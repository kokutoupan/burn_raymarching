use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;

use burn_raymarching::camera::create_camera_rays;
use burn_raymarching::dataset::SceneDataset;
use burn_raymarching::model::scene::SceneModel;
use burn_raymarching::training::{compute_loss, prune_and_split};
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

    let dataset = SceneDataset::new(train_rays_o, train_rays_d, train_targets);
    println!("Total training pixels: {}", dataset.num_total_pixels);
    println!(
        "Foreground pixels: {}, Background pixels: {}",
        dataset.fg_indices.len(),
        dataset.bg_indices.len()
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
            // --- サンプリング比率のアニーリング ---
            let uniform_ratio = 0.8 - (0.6 * progress); // 0.8 -> 0.2 に減少

            // --- バッチサンプリング ---
            let (batch_ro, batch_rd, batch_target) =
                dataset.sample_batch(BATCH_SIZE, uniform_ratio, &device);

            let output = model.forward(batch_ro, batch_rd, smooth_k);

            // ==========================================
            // --- Loss計算 ---
            // ==========================================
            let loss = compute_loss(&model, output, batch_target);

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
        let (next_centers, next_colors, next_radii, next_n) =
            prune_and_split(&model, centers_vec.as_slice(), stage, STAGES);

        // 次世代の情報をセット
        current_n = next_n;
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
