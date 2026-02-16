use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::prelude::*;

use image::{ColorType, save_buffer};

// --- 1. SDF & レンダラー ---
fn sdf_sphere<B: Backend>(p: Tensor<B, 2>, center: Tensor<B, 1>, radius: f32) -> Tensor<B, 2> {
    let diff = p - center.unsqueeze();
    let dist = diff.powf_scalar(2.0).sum_dim(1).sqrt() - radius;
    dist
}

fn render<B: Backend>(
    ray_org: Tensor<B, 2>,
    ray_dir: Tensor<B, 2>,
    sphere_center: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let num_rays = ray_org.dims()[0];
    let device = ray_org.device();
    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    // 32ステップ進める
    for _ in 0..32 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();
        let dist = sdf_sphere(p, sphere_center.clone(), 0.5);
        t = t + dist;
    }

    let p_final = ray_org + ray_dir * t;
    let final_dist = sdf_sphere(p_final, sphere_center, 0.5);

    // シルエット抽出 (Soft Mask)
    (-final_dist.powf_scalar(2.0) * 5.0).exp()
}

// --- 画像保存用ヘルパー関数 ---
fn save_tensor_as_image<B: Backend>(tensor: Tensor<B, 2>, width: u32, height: u32, path: &str) {
    // 1. GPU/BackendからデータをCPUへ (Vec<f32>)
    // [NumRays, 1] -> フラットなVec<f32>
    let floats: Vec<f32> = tensor.into_data().to_vec::<f32>().unwrap();

    // 2. f32 (0.0-1.0) -> u8 (0-255) に変換
    let pixels: Vec<u8> = floats
        .iter()
        .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    // 3. imageクレートで保存
    // グレースケール (L8) として保存
    save_buffer(path, &pixels, width, height, ColorType::L8).expect("Failed to save image");

    println!("Saved image to {}", path);
}

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // --- 画面設定 (64x64 = 4096 pixels) ---
    // これなら「全体像」が見えるので、どこにいても勾配が伝わる
    let width = 64;
    let height = 64;
    let num_rays = width * height;

    // Ray Origin: 全ピクセル (0, 0, -2)
    let ray_org = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, -2.0], &device)
        .unsqueeze()
        .repeat_dim(0, num_rays);

    // Ray Direction: UVグリッド作成
    let mut dirs = Vec::with_capacity(num_rays * 3);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 / width as f32) * 2.0 - 1.0;
            let v = -((y as f32 / height as f32) * 2.0 - 1.0);
            dirs.push(u);
            dirs.push(v);
            dirs.push(1.0);
        }
    }
    let ray_dir =
        Tensor::<MyBackend, 1>::from_floats(dirs.as_slice(), &device).reshape([num_rays, 3]);

    // --- 問題設定 ---
    // Target: 右上にある (0.3, 0.3)
    let target_pos = Tensor::<MyBackend, 1>::from_floats([0.3, 0.3, 0.0], &device);

    // Learner: 左下からスタート (-0.5, -0.5)
    // ※ 1本のレイだと外れて終わるが、画像全体なら「画面の端」として認識される
    let init_pos = Tensor::<MyBackend, 1>::from_floats([-0.5, -0.5, 0.0], &device);
    let mut param_pos = init_pos.clone().require_grad();

    // 正解画像 (Target Image) を作成
    let target_img = render(ray_org.clone(), ray_dir.clone(), target_pos.clone()).detach();

    save_tensor_as_image(
        target_img.clone(),
        width as u32,
        height as u32,
        "target.png",
    );

    println!("Start Optimization with {} rays...", num_rays);
    println!("Target: {}", target_pos);
    println!("Initial: {}", init_pos);

    // --- 学習ループ ---
    let lr = 0.5; // 画像全体でのLossは小さいので、少し学習率を上げてみる

    for i in 0..100 {
        // 1. 描画 (Forward)
        let img = render(ray_org.clone(), ray_dir.clone(), param_pos.clone());

        // 2. 比較 (Loss: Mean Squared Error)
        // 画像同士の引き算になる
        let loss = (img - target_img.clone()).powf_scalar(2.0).mean();

        // 3. 逆伝播 (Backward)
        let grads = loss.backward();

        // 4. 更新
        let grad = param_pos.grad(&grads).unwrap();
        let new_val = param_pos.inner() - grad * lr;
        param_pos = Tensor::from_inner(new_val).require_grad();

        if i % 10 == 0 {
            println!(
                "Step {}: Loss = {:.6}, Pos = {}",
                i,
                loss.into_scalar(),
                param_pos
            );
        }
    }

    println!("Final Result: {}", param_pos);
}
