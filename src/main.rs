use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use image::{ColorType, save_buffer};

// --- 1. モデル定義 (Module) ---
#[derive(Module, Debug)]
struct SceneModel<B: Backend> {
    pos: Param<Tensor<B, 1>>,
    color: Param<Tensor<B, 1>>,
}

impl<B: Backend> SceneModel<B> {
    fn new(pos: Tensor<B, 1>, color: Tensor<B, 1>) -> Self {
        Self {
            pos: Param::from_tensor(pos),
            color: Param::from_tensor(color),
        }
    }

    fn forward(&self, ray_org: Tensor<B, 2>, ray_dir: Tensor<B, 2>) -> Tensor<B, 2> {
        let color_rgb = activation::sigmoid(self.color.val());
        render(ray_org, ray_dir, self.pos.val(), color_rgb)
    }
}

// SDF
fn sdf_sphere<B: Backend>(p: Tensor<B, 2>, center: Tensor<B, 1>, radius: f32) -> Tensor<B, 2> {
    let diff = p - center.unsqueeze();
    (diff.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt() - radius
}

fn calc_normal<B: Backend>(p: Tensor<B, 2>, center: Tensor<B, 1>) -> Tensor<B, 2> {
    let eps = 1e-4;
    let e_x = Tensor::<B, 1>::from_floats([eps, 0.0, 0.0], &p.device()).unsqueeze();
    let e_y = Tensor::<B, 1>::from_floats([0.0, eps, 0.0], &p.device()).unsqueeze();
    let e_z = Tensor::<B, 1>::from_floats([0.0, 0.0, eps], &p.device()).unsqueeze();

    let nx = sdf_sphere(p.clone() + e_x.clone(), center.clone(), 0.5)
        - sdf_sphere(p.clone() - e_x, center.clone(), 0.5);
    let ny = sdf_sphere(p.clone() + e_y.clone(), center.clone(), 0.5)
        - sdf_sphere(p.clone() - e_y, center.clone(), 0.5);
    let nz = sdf_sphere(p.clone() + e_z.clone(), center.clone(), 0.5)
        - sdf_sphere(p.clone() - e_z, center.clone(), 0.5);

    let n = Tensor::cat(vec![nx, ny, nz], 1);
    let len = (n.clone().powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    n / len
}

fn render<B: Backend>(
    ray_org: Tensor<B, 2>,
    ray_dir: Tensor<B, 2>,
    sphere_center: Tensor<B, 1>,
    sphere_color: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let num_rays = ray_org.dims()[0];
    let device = ray_org.device();
    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    for _ in 0..32 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();
        let dist = sdf_sphere(p, sphere_center.clone(), 0.5);
        t = t + dist;
    }

    let p_final = ray_org + ray_dir * t;
    let normal = calc_normal(p_final.clone(), sphere_center.clone());

    let light_dir = Tensor::<B, 1>::from_floats([-0.5, 0.5, -1.0], &device);
    let light_dir = light_dir.clone() / (light_dir.clone().powf_scalar(2.0).sum().sqrt() + 1e-6);
    let light_dir = light_dir.unsqueeze();

    let diffuse = (normal * light_dir).sum_dim(1).clamp_min(0.0);
    let ambient = 0.1;
    let lighting = diffuse + ambient;
    let object_color = sphere_color.unsqueeze() * lighting;

    let final_dist = sdf_sphere(p_final, sphere_center, 0.5);
    let mask = (-final_dist.powf_scalar(2.0) * 5.0).exp();

    object_color * mask
}

fn save_tensor_as_image<B: Backend>(tensor: Tensor<B, 2>, width: u32, height: u32, path: &str) {
    let floats: Vec<f32> = tensor.into_data().to_vec::<f32>().unwrap();
    let pixels: Vec<u8> = floats
        .iter()
        .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    save_buffer(path, &pixels, width, height, ColorType::Rgb8).expect("Failed to save image");
    println!("Saved image to {}", path);
}

fn main() {
    // Autodiffバックエンドの定義
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let width = 256;
    let height = 256;
    let num_rays = width * height;

    // Ray Origin
    let ray_org = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, -2.0], &device)
        .unsqueeze()
        .repeat_dim(0, num_rays);

    // Ray Direction
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

    // Problem Setup
    let target_pos = Tensor::<MyBackend, 1>::from_floats([0.3, 0.3, 0.0], &device);
    let target_color = Tensor::<MyBackend, 1>::from_floats([1.0, 0.0, 0.0], &device);
    let init_pos = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);
    let init_color_logits = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);

    let mut model = SceneModel::new(init_pos, init_color_logits);

    // Optimizer初期化
    let mut optim = AdamConfig::new().init();

    // Target Image
    let target_img = render(
        ray_org.clone(),
        ray_dir.clone(),
        target_pos.clone(),
        target_color.clone(),
    )
    .detach();

    save_tensor_as_image(
        target_img.clone(),
        width as u32,
        height as u32,
        "target.png",
    );

    println!("Start Optimization...");
    println!("Target: {}", target_pos);
    println!("Initial: {}", model.pos.val());

    let lr = 0.1;

    for i in 0..100 {
        let img = model.forward(ray_org.clone(), ray_dir.clone());

        if i % 20 == 0 {
            save_tensor_as_image(
                img.clone(),
                width as u32,
                height as u32,
                &format!("step_{}.png", i),
            );
        }

        let loss = (img - target_img.clone()).powf_scalar(2.0).mean();

        let grads = loss.backward();

        let grads = GradientsParams::from_grads(grads, &model);

        model = optim.step(lr, model, grads);

        if i % 20 == 0 {
            println!(
                "Step {}: Loss = {:.6}\n  Pos: {}\n  Col: {}",
                i,
                loss.into_scalar(),
                model.pos.val(),
                activation::sigmoid(model.color.val()),
            );
        }
    }

    println!(
        "Final Result:\n  Pos: {}\n  Col: {}",
        model.pos.val(),
        activation::sigmoid(model.color.val()),
    );
}
