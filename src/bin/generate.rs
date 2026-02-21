use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::*;
use serde::Serialize;
use std::f32::consts::PI;

use burn_raymarching::camera::create_camera_rays;
use burn_raymarching::renderer::render;
use burn_raymarching::util::save_tensor_as_image;

// JSON出力用のカメラ設定構造体
#[derive(Serialize)]
struct CameraConfig {
    file: String,
    origin: [f32; 3],
    target: [f32; 3],
    fov: f32,
}

fn main() {
    type MyBackend = Wgpu;
    let device = WgpuDevice::default();

    let width = 256;
    let height = 256;

    // --- Target: 3つの球 (三色団子) ---
    // 左(赤), 中(緑), 右(青)
    let target_centers = Tensor::<MyBackend, 1>::from_floats(
        [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
        &device,
    )
    .reshape([3, 3]);

    let target_colors =
        Tensor::<MyBackend, 1>::from_floats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &device)
            .reshape([3, 3]);

    let target_radii =
        Tensor::<MyBackend, 1>::from_floats([0.2, 0.15, 0.2], &device).reshape([3, 1]);

    // --- カメラ位置の生成アルゴリズム ---
    let mut cameras = Vec::new();
    let num_horizontal = 8;
    let radius = 2.5;
    let target_pos = [0.0, 0.0, 0.0];
    let fov = 50.0;

    // 1. 水平方向をぐるっと8方向 (少し上から見下ろすアングル y=0.5)
    for i in 0..num_horizontal {
        // 45度ずつ回転
        let angle = (i as f32) * (2.0 * PI / num_horizontal as f32);
        let cx = radius * angle.cos();
        let cz = radius * angle.sin();
        let cy = 0.5; // 少し上から俯瞰する

        cameras.push(CameraConfig {
            file: format!("data/target_{}.png", i),
            origin: [cx, cy, cz],
            target: target_pos,
            fov,
        });
    }

    // 2. 真上(Top)からの視点を追加
    // ※ y軸と視線が完全に平行になると外積計算がバグるので、zを -0.001 だけズラす
    cameras.push(CameraConfig {
        file: "data/target_8.png".to_string(),
        origin: [0.0, 2.5, -0.001],
        target: target_pos,
        fov,
    });

    // 3. 少し下からの視点(Bottom-ish)を追加
    cameras.push(CameraConfig {
        file: "data/target_9.png".to_string(),
        origin: [0.0, -1.5, -2.0],
        target: target_pos,
        fov,
    });

    // 保存先ディレクトリの作成（無い場合）
    std::fs::create_dir_all("data").ok();

    println!("Generating {} target images...", cameras.len());

    // --- レンダリングと画像の保存 ---
    for cam in &cameras {
        println!("  -> Rendering {}", cam.file);

        let (ray_org, ray_dir) = create_camera_rays::<MyBackend>(
            width, height, cam.origin, cam.target, cam.fov, &device,
        );

        let img = render(
            ray_org,
            ray_dir,
            target_centers.clone(),
            target_colors.clone(),
            target_radii.clone(),
        );

        save_tensor_as_image(img, width as u32, height as u32, &cam.file);
    }

    // --- cameras.json の出力 ---
    let json_file =
        std::fs::File::create("data/cameras.json").expect("Failed to create cameras.json");
    serde_json::to_writer_pretty(json_file, &cameras).expect("Failed to write JSON");

    println!("🎉 Done! All images and data/cameras.json have been generated.");
}
