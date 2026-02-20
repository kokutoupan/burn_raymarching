use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
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
    const N: usize = 20;
    const BATCH_SIZE: usize = 4096; // VRAMに合わせて調整 (2048~8192くらい)
    const ITERATIONS: usize = 2000; // バッチ学習なので回数を増やす

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

    // --- 3. モデル初期化 (100個) ---
    // 初期配置を少し広げる (0.5 -> 0.8)
    let init_centers =
        Tensor::<MyBackend, 1>::random([N, 3], Distribution::Uniform(-0.8, 0.8), &device)
            .reshape([N, 3]);
    let init_logits = Tensor::<MyBackend, 1>::zeros([N * 3], &device).reshape([N, 3]);
    let init_radii = Tensor::<MyBackend, 1>::from_floats([0.0; N], &device).reshape([N, 1]); // 小さめでスタート(Softplus(-2) ≈ 0.12)

    let mut model = SceneModel::new(init_centers, init_logits, init_radii);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
        .init(); // 重み減衰を少し入れると安定するかも

    println!("Start Optimization (N={} Spheres)...", N);
    let lr = 0.05; // 球が多いので学習率は少し調整が必要かも

    for i in 1..=ITERATIONS {
        // --- 4. バッチサンプリング ---
        let mut rng = rand::rng();

        // バッチを「全体ランダム」と「前景ブースト」で半分ずつに分ける
        let uniform_batch_size = BATCH_SIZE / 2;
        let fg_boost_batch_size = BATCH_SIZE - uniform_batch_size;

        let mut batch_indices = Vec::with_capacity(BATCH_SIZE);

        // 1. 全体からのランダム抽出 (背景と前景が「実際の画像の比率」で自然に選ばれる)
        for _ in 0..uniform_batch_size {
            // 0 から num_total_pixels - 1 までの一様乱数
            batch_indices.push(rng.random_range(0..num_total_pixels as i32));
        }

        // 2. 前景からの抽出 (物体の形を早く作るためのブースト)
        if !fg_indices.is_empty() {
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

        // --- Forward & Backward ---
        let output = model.forward(batch_ro, batch_rd);

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
        // 元の「1.0を超えたら罰則」に加えて、常に少しだけ半径を小さくしようとする L1 正則化を足す
        let radius_l1_penalty = radii.clone().abs().mean();

        let r_mask = radii.clone().greater_elem(1.0);
        let radius_large_penalty = Tensor::zeros_like(&radii)
            .mask_where(r_mask, radii.clone().powf_scalar(2.0))
            .mean();

        // ゴミを消すために L1 ペナルティをわずかに効かせる (0.01 くらいから調整)
        loss = loss + radius_large_penalty * 0.1 + radius_l1_penalty * 0.001;

        // [b] 原点引力: 球がバラバラに散らばるのを防ぐ
        let center_penalty = centers.clone().powf_scalar(2.0).mean();
        loss = loss + center_penalty * 0.1; // Billboard対策で少し強めに設定(0.001 -> 0.05)

        // [c] カメラ近接バリア (Billboard Effect対策の要)
        let centers_val = model.centers.val();

        // 中心の原点からの距離: [N, 3] -> [N, 1]
        let dist_from_origin = centers_val.powf_scalar(2.0).sum_dim(1).sqrt();

        // 球の表面が一番外側に張り出す距離 (中心距離 + 半径)
        let max_reach = dist_from_origin + radii;

        // 境界線 1.5 を超えているかどうかのマスク
        let out_of_bounds_mask = max_reach.clone().greater_elem(1.2);

        // はみ出した距離の2乗をペナルティにする (なめらかに勾配が効くように)
        let excess_dist = max_reach.clone() - 1.1;
        let penalty_values = excess_dist.powf_scalar(2.0);

        let camera_proximity_penalty = Tensor::zeros_like(&max_reach)
            .mask_where(out_of_bounds_mask, penalty_values)
            .mean();

        loss = loss + camera_proximity_penalty * 5.0;

        // [d] 反発項: 球同士の重なりを防ぐ (展開公式による高速化版)
        let centers_val = model.centers.val();
        let c_sq_val = centers_val.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
        let c_sq_t = c_sq_val.clone().transpose(); // [1, N]
        let c_dot_c = centers_val.clone().matmul(centers_val.clone().transpose()); // [N, N]

        let dist_sq = c_sq_val + c_sq_t - c_dot_c * 2.0; // [N, N]
        let dist_matrix = dist_sq.clamp_min(1e-6).sqrt(); // [N, N]
        let eye = Tensor::<MyBackend, 2>::eye(N, &device);
        let repulsion_loss = (dist_matrix + eye * 100.0 + 1e-6).powf_scalar(-1.0).mean();
        loss = loss + repulsion_loss * 0.00001; // 極力弱める(0.0002 -> 0.00001)

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        let current_lr = if i < 1000 {
            lr
        } else if i < 2000 {
            lr * 0.2
        } else {
            lr * 0.04
        };
        model = optim.step(current_lr, model, grads);

        // --- ログ出力 & 画像保存 ---
        if i % 100 == 0 {
            println!("Step {}: Loss = {:.6}", i, loss.into_scalar());
        }

        // プレビュー保存 (VRAM溢れ防止のためタイル分割レンダリング)
        if i % 500 == 0 {
            println!("Saving preview...");
            save_tiled_preview(
                &model,
                ro1.clone(),
                rd1.clone(),
                width,
                height,
                &format!("steps/step_{}.png", i),
            );
        }
    }

    // 最終結果の保存
    println!(
        "Final Result:\n  Pos: {}\n  Col: {}\n  Rad: {}",
        model.centers.val(),
        activation::sigmoid(model.colors.val()),
        activation::softplus(model.radius.val(), 1.0),
    );
    println!("Exporting parameters to scene.json...");

    // 1. パラメータを確定値(物理量)に変換して取り出す
    let centers_tensor = model.centers.val();
    let colors_tensor = activation::sigmoid(model.colors.val()); // 色は0~1に
    let radii_tensor = activation::softplus(model.radius.val(), 1.0); // 半径は正の値に

    // CPUに転送
    let centers_vec: Vec<f32> = centers_tensor
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let colors_vec: Vec<f32> = colors_tensor.into_data().convert::<f32>().to_vec().unwrap();
    let radii_vec: Vec<f32> = radii_tensor.into_data().convert::<f32>().to_vec().unwrap();

    // 2. 保存用の構造体を作る
    #[derive(serde::Serialize)]
    struct SceneData {
        num_spheres: usize,
        centers: Vec<f32>, // [x, y, z, x, y, z, ...]
        colors: Vec<f32>,  // [r, g, b, r, g, b, ...]
        radii: Vec<f32>,   // [r, r, ...]
    }

    let data = SceneData {
        num_spheres: N,
        centers: centers_vec,
        colors: colors_vec,
        radii: radii_vec,
    };

    // 3. JSONで保存
    let file = std::fs::File::create("scene.json").expect("Failed to create file");
    serde_json::to_writer_pretty(file, &data).expect("Failed to write json");

    println!("Export done. Run `cargo run --release --bin viewer`");
    // println!("Rendering final images...");
    // save_tiled_preview(&model, ro1, rd1, width, height, "steps/final_1.png");
    // save_tiled_preview(&model, ro2, rd2, width, height, "steps/final_2.png");
    // save_tiled_preview(&model, ro3, rd3, width, height, "steps/final_3.png");
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

        // スライス
        let batch_ro = rays_o.clone().slice([start..end]);
        let batch_rd = rays_d.clone().slice([start..end]);

        // 推論 (勾配不要なので detach してもいいが、Model::forward が tensor を返すので
        // 返り値を detach するのが簡単)
        let out = model.forward(batch_ro, batch_rd).detach();

        outputs.push(out);
        start += chunk_size;
    }

    // 結合して画像に戻す
    let full_img_flat = Tensor::cat(outputs, 0);

    save_tensor_as_image(full_img_flat, width as u32, height as u32, path);
} // マスク計算 (距離が近いほど1.0)
