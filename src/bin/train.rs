use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
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

    let width = 256;
    let height = 256;

    // --- 1. カメラ設定 (3視点) ---
    // Camera 1: 正面
    let (ray_org_1, ray_dir_1) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 0.0, -2.5],
        [0.0, 0.0, 0.0],
        50.0,
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

    // --- Target: Load from Images ---
    println!("Loading target images...");
    let target_img_1 = load_image_as_tensor::<MyBackend>("data/target_1.png", &device).detach();
    let target_img_2 = load_image_as_tensor::<MyBackend>("data/target_2.png", &device).detach();
    let target_img_3 = load_image_as_tensor::<MyBackend>("data/target_3.png", &device).detach();

    // --- Model Init ---
    // 初期値: 中央付近に(N=3)(shape [N,3])
    let init_centers =
        Tensor::<MyBackend, 1>::random([3, 3], Distribution::Uniform(-0.5, 0.5), &device)
            .reshape([3, 3]);

    let init_logits = Tensor::<MyBackend, 1>::zeros([9], &device).reshape([3, 3]);

    let init_radii = Tensor::<MyBackend, 1>::from_floats([-0.5; 3], &device).reshape([3, 1]);

    let mut model = SceneModel::new(init_centers, init_logits, init_radii);
    let mut optim = AdamConfig::new().init();

    println!("Start Optimization (Metaballs)...");
    let lr = 0.2;

    for i in 0..200 {
        let img1 = model.forward(ray_org_1.clone(), ray_dir_1.clone());
        let img2 = model.forward(ray_org_2.clone(), ray_dir_2.clone());
        let img3 = model.forward(ray_org_3.clone(), ray_dir_3.clone());

        if i % 20 == 0 {
            save_tensor_as_image(
                img1.clone(),
                width as u32,
                height as u32,
                &format!("steps/step_{}.png", i),
            );
        }

        let loss = (img1 - target_img_1.clone()).powf_scalar(2.0).mean()
            + (img2 - target_img_2.clone()).powf_scalar(2.0).mean()
            + (img3 - target_img_3.clone()).powf_scalar(2.0).mean();

        let grads = loss.backward();

        let grads = GradientsParams::from_grads(grads, &model);

        model = optim.step(lr, model, grads);

        if i % 20 == 0 {
            println!(
                "Step {}: Loss = {:.6}\n  Pos: {}\n  Col: {}",
                i,
                loss.into_scalar(),
                model.centers.val(),
                activation::sigmoid(model.colors.val()),
            );
        }
    }

    println!(
        "Final Result:\n  Pos: {}\n  Col: {}\n  Rad: {}",
        model.centers.val(),
        activation::sigmoid(model.colors.val()),
        activation::softplus(model.radius.val(), 1.0),
    );

    let final_img1 = model.forward(ray_org_1.clone(), ray_dir_1.clone());
    let final_img2 = model.forward(ray_org_2.clone(), ray_dir_2.clone());
    let final_img3 = model.forward(ray_org_3.clone(), ray_dir_3.clone());

    save_tensor_as_image(
        final_img1.clone(),
        width as u32,
        height as u32,
        "steps/final_1.png",
    );
    save_tensor_as_image(
        final_img2.clone(),
        width as u32,
        height as u32,
        "steps/final_2.png",
    );
    save_tensor_as_image(
        final_img3.clone(),
        width as u32,
        height as u32,
        "steps/final_3.png",
    );
}
