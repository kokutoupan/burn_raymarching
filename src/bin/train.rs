use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation;

use burn_raymarching::camera::create_camera_rays;
use burn_raymarching::model::scene::SceneModel;
use burn_raymarching::util::{load_image_as_tensor, save_tensor_as_image};

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // --------------------------------------------------------
    // 設定: 球を100個に増やす
    // --------------------------------------------------------
    const N: usize = 100;
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
    let train_rays_o = Tensor::cat(vec![ro1.clone(), ro2.clone(), ro3.clone()], 0); // [TotalPixels, 3]
    let train_rays_d = Tensor::cat(vec![rd1.clone(), rd2.clone(), rd3.clone()], 0);
    let train_targets = Tensor::cat(vec![t1, t2, t3], 0);

    let num_total_pixels = train_rays_o.dims()[0];
    println!("Total training pixels: {}", num_total_pixels);

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
        // ランダムなインデックスを生成
        let indices = Tensor::<MyBackend, 1>::random(
            [BATCH_SIZE],
            Distribution::Uniform(0.0, num_total_pixels as f64),
            &device,
        )
        .int(); // float -> int

        // インデックスを使ってデータを抽出 (Gather)
        let batch_ro = train_rays_o.clone().select(0, indices.clone());
        let batch_rd = train_rays_d.clone().select(0, indices.clone());
        let batch_target = train_targets.clone().select(0, indices);

        // --- Forward & Backward ---
        let output = model.forward(batch_ro, batch_rd);

        // --- Loss計算 ---
        let diff = output - batch_target.clone();
        let mse_map = diff.clone().powf_scalar(2.0);
        let abs_diff = diff.abs();

        // 1. 物体領域(黒くない画素)の判定
        let target_sum = batch_target.sum_dim(1);
        let target_mask = target_sum.greater_elem(0.01); // [Batch, 1] Bool

        // 2. 混合Loss (物体領域は L1 で厳しく、背景は MSE)
        let mixed_loss_map = mse_map.mask_where(
            target_mask,     // condition
            abs_diff * 10.0, // replacement (L1 * 10)
        );
        let mut loss = mixed_loss_map.mean();

        // 3. 反発項 (0除算/無限勾配回避)
        let centers = model.centers.val();
        let c1 = centers.clone().unsqueeze_dim::<3>(1);
        let c2 = centers.clone().unsqueeze_dim::<3>(0);
        let dist_sq = (c1 - c2).powf_scalar(2.0).sum_dim(2);
        let dist_matrix = (dist_sq + 1e-6).sqrt().squeeze_dim(2);
        let eye = Tensor::<MyBackend, 2>::eye(N, &device);
        let repulsion_loss = (dist_matrix + eye * 100.0 + 1e-6).powf_scalar(-1.0).mean();
        loss = loss + repulsion_loss * 0.0002;

        // 4. 半径の巨大化ペナルティ (型エラー修正版)
        let radius_val = activation::softplus(model.radius.val(), 1.0);
        let r_mask = radius_val.clone().greater_elem(1.0); // 1.0を超えたら罰則
        let radius_penalty = Tensor::zeros_like(&radius_val)
            .mask_where(r_mask, radius_val.powf_scalar(2.0))
            .mean();
        loss = loss + radius_penalty * 0.1;

        // 5. 画面外への逃亡防止
        let center_penalty = model.centers.val().powf_scalar(2.0).mean();
        loss = loss + center_penalty * 0.001;
        // let loss = mse_loss;

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
