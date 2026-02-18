use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::prelude::*;
use burn_raymarching::camera::create_camera_rays;
use burn_raymarching::renderer::render;
use burn_raymarching::util::save_tensor_as_image;

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let width = 256;
    let height = 256;

    // --- 1. カメラ設定 (3視点) ---
    // Camera 1: 正面
    let (ray_org_1, ray_dir_1) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 0.0, -2.5], // Eye
        [0.0, 0.0, 0.0],  // Target
        50.0,             // FOV
        &device,
    );

    // Camera 2: 真横
    let (ray_org_2, ray_dir_2) = create_camera_rays::<MyBackend>(
        width,
        height,
        [2.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );

    // Camera 3: 真上
    let (ray_org_3, ray_dir_3) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 2.5, -0.001],
        [0.0, 0.0, 0.0],
        50.0,
        &device,
    );

    // --- Target: 3つの球 ---
    let target_centers = Tensor::<MyBackend, 1>::from_floats(
        [
            -0.3, 0.0, 0.0, // 左
            0.0, 0.0, 0.0, // 中
            0.3, 0.0, 0.0,
        ], // 右
        &device,
    )
    .reshape([3, 3]);

    let target_colors =
        Tensor::<MyBackend, 1>::from_floats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &device)
            .reshape([3, 3]);

    let target_radii =
        Tensor::<MyBackend, 1>::from_floats([0.2, 0.15, 0.2], &device).reshape([3, 1]);

    println!("Generating target images...");

    // Render and Save
    let target_img_1 = render(
        ray_org_1.clone(),
        ray_dir_1.clone(),
        target_centers.clone(),
        target_colors.clone(),
        target_radii.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img_1,
        width as u32,
        height as u32,
        "data/target_1.png",
    );

    let target_img_2 = render(
        ray_org_2.clone(),
        ray_dir_2.clone(),
        target_centers.clone(),
        target_colors.clone(),
        target_radii.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img_2,
        width as u32,
        height as u32,
        "data/target_2.png",
    );

    let target_img_3 = render(
        ray_org_3.clone(),
        ray_dir_3.clone(),
        target_centers.clone(),
        target_colors.clone(),
        target_radii.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img_3,
        width as u32,
        height as u32,
        "data/target_3.png",
    );

    println!("Done.");
}
