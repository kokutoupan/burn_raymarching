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
use serde::Deserialize;

#[derive(Deserialize)]
struct CameraConfig {
    file: String,
    origin: [f32; 3],
    target: [f32; 3],
    fov: f32,
}

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // --------------------------------------------------------
    // 設定: 球を100個に増やす
    // --------------------------------------------------------
    const BATCH_SIZE: usize = 16384; // VRAMに合わせて調整 (2048~8192くらい)

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
    let (_ro2, _rd2) = create_camera_rays::<MyBackend>(
        width,
        height,
        [2.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );
    let (_ro3, _rd3) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 2.5, -0.0001],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );

    const JSON_PATH: &str = "data/cameras.json";

    // --- JSONからカメラと画像を動的にロード ---
    println!("Loading camera configurations...");
    let config_str = std::fs::read_to_string(JSON_PATH).expect("Failed to read cameras.json");
    let cameras: Vec<CameraConfig> =
        serde_json::from_str(&config_str).expect("Failed to parse JSON");

    let mut all_rays_o = Vec::new();
    let mut all_rays_d = Vec::new();
    let mut all_targets = Vec::new();

    for cam in &cameras {
        let (ro, rd) = create_camera_rays::<MyBackend>(
            width, height, cam.origin, cam.target, cam.fov, &device,
        );
        let target_img = load_image_as_tensor::<MyBackend>(&cam.file, &device)
            .reshape([-1, 3])
            .detach();

        all_rays_o.push(ro);
        all_rays_d.push(rd);
        all_targets.push(target_img);
    }

    // 結合
    let train_rays_o = Tensor::cat(all_rays_o, 0).detach();
    let train_rays_d = Tensor::cat(all_rays_d, 0).detach();
    let train_targets = Tensor::cat(all_targets, 0).detach();

    let dataset = SceneDataset::new(train_rays_o, train_rays_d, train_targets);
    println!("Total training pixels: {}", dataset.num_total_pixels);
    println!(
        "Foreground pixels: {}, Background pixels: {}",
        dataset.fg_indices.len(),
        dataset.bg_indices.len()
    );

    // ==========================================
    // 1. 初期設定 (1個からスタート)
    // ==========================================
    let mut current_n = 7;
    let mut centers_vec = vec![0.0; current_n * 3];
    let mut colors_vec = vec![0.0; current_n * 3]; // Logit 0.0 (グレー)
    let mut radii_vec = vec![0.0; current_n]; // Softplus(-2.0) ≒ 0.12
    let mut light_dir_vec: Vec<f32> = vec![0.0, 1.0, 0.0];
    let mut ambient_intensity_vec: Vec<f32> = vec![-1.4]; // sigmoid(-1.4) ≒ 0.2

    // 6方向 + 中心に球を配置
    let directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for i in 0..6 {
        centers_vec[i * 3] = directions[i][0] * 0.1;
        centers_vec[i * 3 + 1] = directions[i][1] * 0.1;
        centers_vec[i * 3 + 2] = directions[i][2] * 0.1;
    }
    centers_vec[6 * 3] = 0.0;
    centers_vec[6 * 3 + 1] = 0.0;
    centers_vec[6 * 3 + 2] = 0.0;

    const STAGES: usize = 5; // 世代数 (例: 5世代)
    const STEPS_PER_STAGE: usize = 700; // 1世代あたりの学習回数
    const TOTAL_STEPS: f32 = (STAGES * STEPS_PER_STAGE) as f32;
    const MAX_SMOOTH: f32 = 32.0;

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
        let init_light_dir = Tensor::<MyBackend, 1>::from_floats(light_dir_vec.as_slice(), &device);
        let init_ambient_intensity =
            Tensor::<MyBackend, 1>::from_floats(ambient_intensity_vec.as_slice(), &device);

        let mut model = SceneModel::new(
            init_centers,
            init_colors,
            init_radii,
            init_light_dir,
            init_ambient_intensity,
        );

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
            let smooth_k = 5.0 + (MAX_SMOOTH - 5.0) * progress;
            // --- サンプリング比率のアニーリング ---
            let uniform_ratio = 0.8 - (0.4 * progress); // 0.8 -> 0.4 に減少

            // --- バッチサンプリング ---
            let (batch_ro, batch_rd, batch_target) =
                dataset.sample_batch(BATCH_SIZE, uniform_ratio, &device);

            let output = model.forward(batch_ro, batch_rd, smooth_k);

            // ==========================================
            // --- Loss計算 ---
            // ==========================================
            let loss = compute_loss(&model, output, batch_target, progress);

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
            let final_light_dir: Vec<f32> = model
                .light_dir
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            let ambient_tensor = activation::sigmoid(model.ambient_intensity.val());
            let final_ambient_intensity: Vec<f32> = ambient_tensor
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();

            // 2. JSONへの保存
            #[derive(serde::Serialize)]
            struct SceneData {
                num_spheres: usize,
                centers: Vec<f32>,
                colors: Vec<f32>,
                radii: Vec<f32>,
                light_dir: Vec<f32>,
                ambient_intensity: Vec<f32>,
            }

            let data = SceneData {
                num_spheres: current_n,
                centers: final_centers,
                colors: final_colors,
                radii: final_radii,
                light_dir: final_light_dir,
                ambient_intensity: final_ambient_intensity,
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
            // save_tiled_preview(
            //     &model,
            //     ro2.clone(),
            //     rd2.clone(),
            //     width,
            //     height,
            //     "steps/final_2.png",
            // );
            // save_tiled_preview(
            //     &model,
            //     ro3.clone(),
            //     rd3.clone(),
            //     width,
            //     height,
            //     "steps/final_3.png",
            // );

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
        light_dir_vec = model
            .light_dir
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        ambient_intensity_vec = model
            .ambient_intensity
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
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
        let current_chunk_size = out.dims()[0];
        let out_color = out.slice([0..current_chunk_size, 0..3]);
        outputs.push(out_color);
        start += chunk_size;
    }

    // 結合して画像に戻す
    let full_img_flat = Tensor::cat(outputs, 0);

    save_tensor_as_image(full_img_flat, width as u32, height as u32, path);
}
