use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;

use rand::RngExt;

use burn_raymarching::camera::create_camera_rays;
use burn_raymarching::model::scene::SceneModel;
use burn_raymarching::util::{load_image_as_tensor, save_tensor_as_image};

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // --------------------------------------------------------
    // 設定: 球を100個に増やす
    // --------------------------------------------------------
    const BATCH_SIZE: usize = 8192; // VRAMに合わせて調整 (2048~8192くらい)

    let width = 256;
    let height = 256;

    // --- 1. カメラと正解画像の準備 ---
    // 各視点のレイ生成 (戻り値は [H*W, 3])
    let (ro1, rd1) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 0.0, -2.5],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );
    let (ro2, rd2) = create_camera_rays::<MyBackend>(
        width,
        height,
        [2.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );
    let (ro3, rd3) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 2.5, -0.0001],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );

    // 正解画像のロード (3次元 [H, W, 3] で来るので フラット [H*W, 3] にする)
    let t1 = load_image_as_tensor::<MyBackend>("data/target_1.png", &device)
        .reshape([-1, 3])
        .detach();
    let t2 = load_image_as_tensor::<MyBackend>("data/target_2.png", &device)
        .reshape([-1, 3])
        .detach();
    let t3 = load_image_as_tensor::<MyBackend>("data/target_3.png", &device)
        .reshape([-1, 3])
        .detach();

    // --- 2. データセットの統合 ---
    // 全ての視点のデータを結合して、巨大な「学習用プール」を作る
    let train_rays_o = Tensor::cat(vec![ro1.clone(), ro2.clone(), ro3.clone()], 0).detach(); // [TotalPixels, 3]
    let train_rays_d = Tensor::cat(vec![rd1.clone(), rd2.clone(), rd3.clone()], 0).detach();
    let train_targets = Tensor::cat(vec![t1, t2, t3], 0).detach();

    let num_total_pixels = train_rays_o.dims()[0];
    println!("Total training pixels: {}", num_total_pixels);

    // --- 2.5 サンプリング効率化のためのインデックス事前計算 (CPU側) ---
    let targets_data: Vec<f32> = train_targets
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let mut fg_indices = Vec::new();
    let mut bg_indices = Vec::new();

    for i in 0..num_total_pixels {
        let idx = i * 3;
        let sum_color = targets_data[idx] + targets_data[idx + 1] + targets_data[idx + 2];

        // 色の合計が一定以上なら物体（前景）、そうでなければ背景
        if sum_color > 0.05 {
            fg_indices.push(i as i32);
        } else {
            bg_indices.push(i as i32);
        }
    }
    println!(
        "Foreground pixels: {}, Background pixels: {}",
        fg_indices.len(),
        bg_indices.len()
    );

    // ==========================================
    // 1. 初期設定 (最初は5個からスタート)
    // ==========================================
    let mut current_n = 5;
    let mut centers_vec = vec![0.0; current_n * 3];
    let mut colors_vec = vec![0.0; current_n * 3]; // Logit 0.0 (グレー)
    let mut radii_vec = vec![0.0; current_n]; // Softplus(-2.0) ≒ 0.12

    // 初期位置を少しだけ散らす
    for i in 0..current_n {
        centers_vec[i * 3 + 0] = (i as f32 * 0.1) - 0.2;
    }

    const STAGES: usize = 5; // 世代数 (例: 5世代)
    const STEPS_PER_STAGE: usize = 600; // 1世代あたりの学習回数
    const TOTAL_STEPS: f32 = (STAGES * STEPS_PER_STAGE) as f32;

    println!("🚀 Start Multi-Stage Optimization...");

    // ==========================================
    // 2. 世代（Stage）ループ
    // ==========================================
    for stage in 0..STAGES {
        println!("=== Stage {}/{} (N = {}) ===", stage + 1, STAGES, current_n);

        // --- A. モデルとOptimizerの再構築 ---
        let init_centers = Tensor::<MyBackend, 1>::from_floats(centers_vec.as_slice(), &device)
            .reshape([current_n, 3]);
        let init_colors = Tensor::<MyBackend, 1>::from_floats(colors_vec.as_slice(), &device)
            .reshape([current_n, 3]);
        let init_radii = Tensor::<MyBackend, 1>::from_floats(radii_vec.as_slice(), &device)
            .reshape([current_n, 1]);

        let mut model = SceneModel::new(init_centers, init_colors, init_radii);

        // ★重要: ステージごとにAdamを作り直す（古いテンソルサイズのモメンタムをリセットして爆発を防ぐ）
        let mut optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
            .init();

        // 学習率もステージが進むにつれて少しずつ下げる
        let base_lr = 0.05 * (0.6f64).powi(stage as i32);

        // --- B. 1世代分の学習ループ ---
        for step in 1..=STEPS_PER_STAGE {
            let global_step = (stage * STEPS_PER_STAGE + step) as f32;
            let progress = global_step / TOTAL_STEPS;

            // [ここで先ほどの「サンプリング比率のアニーリング」と「kのアニーリング」を行う]
            let smooth_k = 5.0 + (32.0 - 5.0) * progress;
            // --- バッチサンプリング ---
            let mut rng = rand::rng();

            // --- サンプリング比率のアニーリング ---
            let uniform_ratio = 0.8 - (0.6 * progress); // 0.8 -> 0.2 に減少

            let mut uniform_batch_size = (BATCH_SIZE as f32 * uniform_ratio) as usize;
            let mut fg_boost_batch_size = BATCH_SIZE - uniform_batch_size;

            // 前景ピクセル数が少ない場合のキャップ処理
            if !fg_indices.is_empty() && fg_indices.len() < fg_boost_batch_size {
                fg_boost_batch_size = fg_indices.len();
                uniform_batch_size = BATCH_SIZE - fg_boost_batch_size;
            }

            let mut batch_indices = Vec::with_capacity(BATCH_SIZE);

            // 1. 全体からの抽出
            for _ in 0..uniform_batch_size {
                batch_indices.push(rng.random_range(0..num_total_pixels as i32));
            }

            // 2. 前景からの抽出
            if !fg_indices.is_empty() && fg_boost_batch_size > 0 {
                for _ in 0..fg_boost_batch_size {
                    let idx = rng.random_range(0..fg_indices.len());
                    batch_indices.push(fg_indices[idx]);
                }
            }

            // Int型のTensorとして生成
            let indices = Tensor::<MyBackend, 1, Int>::from_ints(batch_indices.as_slice(), &device);

            // インデックスを使ってデータを抽出 (Gather)
            let batch_ro = train_rays_o.clone().select(0, indices.clone()).detach();
            let batch_rd = train_rays_d.clone().select(0, indices.clone()).detach();
            let batch_target = train_targets.clone().select(0, indices).detach();

            let output = model.forward(batch_ro, batch_rd, smooth_k);

            // ==========================================
            // --- Loss計算 ---
            // ==========================================

            // 1. 画像再構成Loss (メインの学習目標)
            // ------------------------------------------
            let diff = output - batch_target.clone();
            let mse_map = diff.clone().powf_scalar(2.0);
            let abs_diff = diff.abs();

            let target_sum = batch_target.sum_dim(1);
            let target_mask = target_sum.greater_elem(0.01); // 物体領域の判定

            // 物体領域は厳しく(L1 * 10)、背景は緩く(MSE)
            let reconstruction_loss = mse_map.mask_where(target_mask, abs_diff * 10.0).mean();
            let mut loss = reconstruction_loss;

            // 2. 幾何学的制約 (ペナルティ項)
            // ------------------------------------------
            let centers = model.centers.val();
            let radii = activation::softplus(model.radius.val(), 1.0);

            // [a] 半径ペナルティ: 球が大きくなりすぎるのを防ぎつつ、不要な球を小さくする
            let radius_l1_penalty = radii.clone().abs().mean();

            let r_mask = radii.clone().greater_elem(1.0);
            let radius_large_penalty = Tensor::zeros_like(&radii)
                .mask_where(r_mask, radii.clone().powf_scalar(2.0))
                .mean();

            loss = loss + radius_large_penalty * 0.1 + radius_l1_penalty * 0.004;

            // [b] 原点引力: 球がバラバラに散らばるのを防ぐ
            let center_penalty = centers.clone().powf_scalar(2.0).mean();
            loss = loss + center_penalty * 0.1; // Billboard対策で少し強めに設定

            // [c] カメラ近接バリア (Billboard Effect対策の要)
            let centers_val = model.centers.val();

            // 中心の原点からの距離: [N, 3] -> [N, 1]
            let dist_from_origin = (centers_val.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();

            // 球の表面が一番外側に張り出す距離 (中心距離 + 半径)
            let max_reach = dist_from_origin + radii;

            // 境界線 1.2 を超えているかどうかのマスク
            let out_of_bounds_mask = max_reach.clone().greater_elem(1.2);
            let excess_dist = max_reach.clone() - 1.1;
            let penalty_values = excess_dist.powf_scalar(2.0);

            let camera_proximity_penalty = Tensor::zeros_like(&max_reach)
                .mask_where(out_of_bounds_mask, penalty_values)
                .mean();

            loss = loss + camera_proximity_penalty * 5.0;

            // [d] 反発項: 球同士の重なりを防ぐ
            let centers_val = model.centers.val();
            let c_sq_val = centers_val.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
            let c_sq_t = c_sq_val.clone().transpose(); // [1, N]
            let c_dot_c = centers_val.clone().matmul(centers_val.clone().transpose()); // [N, N]

            let dist_sq = c_sq_val + c_sq_t - c_dot_c * 2.0; // [N, N]
            let dist_matrix = dist_sq.clamp_min(1e-6).sqrt(); // [N, N]
            let eye = Tensor::<MyBackend, 2>::eye(current_n, &device);
            let repulsion_loss = (dist_matrix + eye * 100.0 + 1e-6).powf_scalar(-1.0).mean();
            loss = loss + repulsion_loss * 0.00001;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // ステージ内でも後半は学習率を下げる
            let current_lr = if step > STEPS_PER_STAGE / 2 {
                base_lr * 0.2
            } else {
                base_lr
            };
            model = optim.step(current_lr, model, grads);

            if step % 100 == 0 {
                println!(
                    "  Step {} | Loss: {:.5} | k: {:.1}",
                    step,
                    loss.into_scalar(),
                    smooth_k
                );
            }
        }

        if stage == STAGES - 1 {
            println!("🎉 Final Stage Complete! Exporting results...");

            // 1. 物理パラメータ（確定値）の取り出し
            let centers_tensor = model.centers.val();
            let colors_tensor = activation::sigmoid(model.colors.val()); // 色を0~1に
            let radii_tensor = activation::softplus(model.radius.val(), 1.0); // 半径を正の値に

            let final_centers: Vec<f32> = centers_tensor
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            let final_colors: Vec<f32> =
                colors_tensor.into_data().convert::<f32>().to_vec().unwrap();
            let final_radii: Vec<f32> = radii_tensor.into_data().convert::<f32>().to_vec().unwrap();

            // 2. JSONへの保存
            #[derive(serde::Serialize)]
            struct SceneData {
                num_spheres: usize,
                centers: Vec<f32>,
                colors: Vec<f32>,
                radii: Vec<f32>,
            }

            let data = SceneData {
                num_spheres: current_n,
                centers: final_centers,
                colors: final_colors,
                radii: final_radii,
            };

            let file = std::fs::File::create("scene.json").expect("Failed to create file");
            serde_json::to_writer_pretty(file, &data).expect("Failed to write json");
            println!("  => Saved to scene.json (N = {})", current_n);

            // 3. 最終レンダリング画像の保存
            println!("  => Rendering final images...");
            // （ro1, rd1 などがこのスコープで取れるならそのまま渡す）
            save_tiled_preview(
                &model,
                ro1.clone(),
                rd1.clone(),
                width,
                height,
                "steps/final_1.png",
            );
            save_tiled_preview(
                &model,
                ro2.clone(),
                rd2.clone(),
                width,
                height,
                "steps/final_2.png",
            );
            save_tiled_preview(
                &model,
                ro3.clone(),
                rd3.clone(),
                width,
                height,
                "steps/final_3.png",
            );

            // 全て完了したのでループを抜ける
            break;
        }

        save_tiled_preview(
            &model,
            ro1.clone(),
            rd1.clone(),
            width,
            height,
            &format!("steps/stage_{stage}.png"),
        );

        // --- C. 世代交代フェーズ: Pruning (削除) & Splitting (分裂) ---
        let out_centers: Vec<f32> = model
            .centers
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let out_colors: Vec<f32> = model
            .colors
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let eval_radii: Vec<f32> = activation::softplus(model.radius.val(), 1.0)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut next_centers = Vec::new();
        let mut next_colors = Vec::new();
        let mut next_radii = Vec::new();

        let mut mut_rng = rand::rng();

        for i in 0..current_n {
            let r = eval_radii[i];
            let cx = out_centers[i * 3];
            let cy = out_centers[i * 3 + 1];
            let cz = out_centers[i * 3 + 2];
            let cr = out_colors[i * 3];
            let cg = out_colors[i * 3 + 1];
            let cb = out_colors[i * 3 + 2];

            // 1. Pruning (削除): デカすぎる影(0.25超え)や、極小のゴミ(0.01未満)は次世代に引き継がない
            if r > 0.25 || r < 0.01 {
                continue;
            }

            // 2. Keep (維持): 生存した球はそのまま次世代へ
            next_centers.extend_from_slice(&[cx, cy, cz]);
            next_colors.extend_from_slice(&[cr, cg, cb]);
            next_radii.push(-2.5); // 半径は少し小さめに再初期化

            // 3. Splitting (分裂): 最終ステージ以外なら、生き残った球を分裂させて表現力を上げる
            if stage < STAGES - 1 {
                // 分裂先は、元の位置からわずかにノイズを乗せてズラす
                let offset_x = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;
                let offset_y = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;
                let offset_z = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;

                next_centers.extend_from_slice(&[cx + offset_x, cy + offset_y, cz + offset_z]);
                next_colors.extend_from_slice(&[cr, cg, cb]); // 色は引き継ぐ
                next_radii.push(-2.5);
            }
        }

        // 次世代の情報をセット
        current_n = next_radii.len();
        centers_vec = next_centers;
        colors_vec = next_colors;
        radii_vec = next_radii;
        println!("  => Pruning & Splitting complete. Next N = {}", current_n);
    }
}

// --- ヘルパー: タイル分割レンダリング (VRAM節約) ---
// Autodiffバックエンドのまま推論するとグラフが作られて重いので、
// 必要なら .detach() したり、小さなチャンクに分けて処理する
fn save_tiled_preview<B: Backend>(
    model: &SceneModel<B>,
    rays_o: Tensor<B, 2>, // [H*W, 3]
    rays_d: Tensor<B, 2>,
    width: usize,
    height: usize,
    path: &str,
) {
    let num_pixels = width * height;
    let chunk_size = 4096; // 推論時のバッチサイズ
    let mut outputs = Vec::new();

    let mut start = 0;
    while start < num_pixels {
        let end = (start + chunk_size).min(num_pixels);
        let batch_ro = rays_o.clone().slice([start..end]);
        let batch_rd = rays_d.clone().slice([start..end]);

        // 推論 (勾配不要なので detach してもいいが、Model::forward が tensor を返すので
        // 返り値を detach するのが簡単)
        let out = model.forward(batch_ro, batch_rd, 32.0).detach();
        outputs.push(out);
        start += chunk_size;
    }

    // 結合して画像に戻す
    let full_img_flat = Tensor::cat(outputs, 0);

    save_tensor_as_image(full_img_flat, width as u32, height as u32, path);
}
