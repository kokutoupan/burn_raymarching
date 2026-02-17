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
    pos1: Param<Tensor<B, 1>>,
    color1: Param<Tensor<B, 1>>, // Logits

    pos2: Param<Tensor<B, 1>>,
    color2: Param<Tensor<B, 1>>, // Logits
}

impl<B: Backend> SceneModel<B> {
    fn new(pos1: Tensor<B, 1>, col1: Tensor<B, 1>, pos2: Tensor<B, 1>, col2: Tensor<B, 1>) -> Self {
        Self {
            pos1: Param::from_tensor(pos1),
            color1: Param::from_tensor(col1),
            pos2: Param::from_tensor(pos2),
            color2: Param::from_tensor(col2),
        }
    }

    fn forward(&self, ray_org: Tensor<B, 2>, ray_dir: Tensor<B, 2>) -> Tensor<B, 2> {
        let c1 = activation::sigmoid(self.color1.val());
        let c2 = activation::sigmoid(self.color2.val());

        render(ray_org, ray_dir, self.pos1.val(), c1, self.pos2.val(), c2)
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

// k: 溶け具合 (0.1〜0.5くらい)
fn smooth_min<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, k: f32) -> Tensor<B, 2> {
    // h = max(k - |a - b|, 0) / k
    let h = (Tensor::<B, 1>::from_floats([k], &a.device()).unsqueeze()
        - (a.clone() - b.clone()).abs())
    .clamp_min(0.0)
        / k;

    // result = min(a, b) - h^2 * k / 4
    let min_ab = a.min_pair(b);
    min_ab - h.powf_scalar(2.0) * (k * 0.25)
}

fn calc_normal_scene<B: Backend>(
    p: Tensor<B, 2>,
    c1: Tensor<B, 1>,
    c2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let eps = 1e-4;
    let e_x = Tensor::<B, 1>::from_floats([eps, 0.0, 0.0], &p.device()).unsqueeze();
    let e_y = Tensor::<B, 1>::from_floats([0.0, eps, 0.0], &p.device()).unsqueeze();
    let e_z = Tensor::<B, 1>::from_floats([0.0, 0.0, eps], &p.device()).unsqueeze();

    // シーン全体のSDFを計算するクロージャ的ヘルパー
    let get_dist = |pos: Tensor<B, 2>| -> Tensor<B, 2> {
        let d1 = sdf_sphere(pos.clone(), c1.clone(), 0.4);
        let d2 = sdf_sphere(pos.clone(), c2.clone(), 0.4);
        smooth_min(d1, d2, 0.2) // k=0.2 で融合
    };

    let nx = get_dist(p.clone() + e_x.clone()) - get_dist(p.clone() - e_x);
    let ny = get_dist(p.clone() + e_y.clone()) - get_dist(p.clone() - e_y);
    let nz = get_dist(p.clone() + e_z.clone()) - get_dist(p.clone() - e_z);

    let n = Tensor::cat(vec![nx, ny, nz], 1);
    let len = (n.clone().powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    n / len
}

fn render<B: Backend>(
    ray_org: Tensor<B, 2>,
    ray_dir: Tensor<B, 2>,
    pos1: Tensor<B, 1>,
    col1: Tensor<B, 1>,
    pos2: Tensor<B, 1>,
    col2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let num_rays = ray_org.dims()[0];
    let device = ray_org.device();
    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    // Ray Marching Loop
    for _ in 0..40 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();

        let d1 = sdf_sphere(p.clone(), pos1.clone(), 0.4);
        let d2 = sdf_sphere(p.clone(), pos2.clone(), 0.4);

        // シーン全体の距離
        let dist = smooth_min(d1, d2, 0.2);
        t = t + dist;
    }

    let p_final = ray_org + ray_dir * t;

    // 法線 (シーン全体)
    let normal = calc_normal_scene(p_final.clone(), pos1.clone(), pos2.clone());

    // ライティング
    let light_dir = Tensor::<B, 1>::from_floats([-0.5, 0.5, -1.0], &device);
    let light_dir = light_dir.clone() / (light_dir.powf_scalar(2.0).sum().sqrt() + 1e-6);
    let diffuse = (normal * light_dir.unsqueeze()).sum_dim(1).clamp_min(0.0);
    let lighting = diffuse + 0.1;

    // --- 色の混合  ---
    let d1_final = sdf_sphere(p_final.clone(), pos1.clone(), 0.4); // 負の値になりうる
    let d2_final = sdf_sphere(p_final.clone(), pos2.clone(), 0.4);

    let w1 = (d1_final.clone().neg() * 10.0).exp();
    let w2 = (d2_final.clone().neg() * 10.0).exp();
    let w_sum = w1.clone() + w2.clone() + 1e-5;

    // 混合色 = (c1 * w1 + c2 * w2) / (w1 + w2)
    let mixed_color = (col1.unsqueeze() * w1 + col2.unsqueeze() * w2) / w_sum;

    let object_color = mixed_color * lighting;

    // マスク
    let dist_scene = smooth_min(d1_final, d2_final, 0.2);
    let mask = (-dist_scene.powf_scalar(2.0) * 10.0).exp();

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

    // --- Target Image Generation ---
    // 正解: 「赤(右上)」と「青(左下)」が融合している状態
    let t_pos1 = Tensor::<MyBackend, 1>::from_floats([0.2, 0.2, 0.0], &device);
    let t_col1 = Tensor::<MyBackend, 1>::from_floats([1.0, 0.0, 0.0], &device); // 赤
    let t_pos2 = Tensor::<MyBackend, 1>::from_floats([-0.2, -0.2, 0.0], &device);
    let t_col2 = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 1.0], &device); // 青

    // 正解画像作成 (モデルを使わず直接関数を呼ぶ)
    let target_img = render(
        ray_org.clone(),
        ray_dir.clone(),
        t_pos1.clone(),
        t_col1.clone(),
        t_pos2.clone(),
        t_col2.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img.clone(),
        width as u32,
        height as u32,
        "target.png",
    );

    // --- Model Init ---
    // 初期状態: 2つとも真ん中付近で、色はグレー
    let init_pos1 = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);
    let init_col1 = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device); // Logit 0 -> Sigmoid 0.5
    let init_pos2 = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);
    let init_col2 = Tensor::<MyBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);

    let mut model = SceneModel::new(init_pos1, init_col1, init_pos2, init_col2);
    let mut optim = AdamConfig::new().init();

    println!("Start Optimization (Metaballs)...");
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
                model.pos1.val(),
                activation::sigmoid(model.color1.val()),
            );
        }
    }

    println!(
        "Final Result:\n  Pos1: {}\n  Col1: {}\n  Pos2: {}\n  Col2: {}",
        model.pos1.val(),
        activation::sigmoid(model.color1.val()),
        model.pos2.val(),
        activation::sigmoid(model.color2.val()),
    );
    let img = model.forward(ray_org.clone(), ray_dir.clone());
    save_tensor_as_image(img.clone(), width as u32, height as u32, "final.png");
}
