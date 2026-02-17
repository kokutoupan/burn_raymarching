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
    centers: Param<Tensor<B, 2>>,
    colors: Param<Tensor<B, 2>>, // Logits
}

impl<B: Backend> SceneModel<B> {
    fn new(centers: Tensor<B, 2>, colors: Tensor<B, 2>) -> Self {
        Self {
            centers: Param::from_tensor(centers),
            colors: Param::from_tensor(colors),
        }
    }

    fn forward(&self, ray_org: Tensor<B, 2>, ray_dir: Tensor<B, 2>) -> Tensor<B, 2> {
        let colors_rgb = activation::sigmoid(self.colors.val());
        let centers = self.centers.val();

        render(ray_org, ray_dir, centers, colors_rgb)
    }
}

// SDF
fn sdf_sphere<B: Backend>(p: Tensor<B, 2>, center: Tensor<B, 1>, radius: f32) -> Tensor<B, 2> {
    let diff = p - center.unsqueeze();
    (diff.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt() - radius
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

fn calc_normal_scene<B: Backend>(p: Tensor<B, 2>, centers: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-4;
    let e_x = Tensor::<B, 1>::from_floats([eps, 0.0, 0.0], &p.device()).unsqueeze();
    let e_y = Tensor::<B, 1>::from_floats([0.0, eps, 0.0], &p.device()).unsqueeze();
    let e_z = Tensor::<B, 1>::from_floats([0.0, 0.0, eps], &p.device()).unsqueeze();

    // シーン全体のSDFを計算するクロージャ的ヘルパー
    let get_dist = |pos: Tensor<B, 2>| -> Tensor<B, 2> { scene_sdf_value(pos, centers.clone()) };

    let nx = get_dist(p.clone() + e_x.clone()) - get_dist(p.clone() - e_x);
    let ny = get_dist(p.clone() + e_y.clone()) - get_dist(p.clone() - e_y);
    let nz = get_dist(p.clone() + e_z.clone()) - get_dist(p.clone() - e_z);

    let n = Tensor::cat(vec![nx, ny, nz], 1);
    let len = (n.clone().powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    n / len
}

// 配列からSDF値を計算する関数 (ループ処理)
fn scene_sdf_value<B: Backend>(p: Tensor<B, 2>, centers: Tensor<B, 2>) -> Tensor<B, 2> {
    let num_spheres = centers.dims()[0];

    // 最初の球の距離で初期化
    let first_center = centers.clone().slice([0..1]).squeeze::<1>();
    let mut min_dist = sdf_sphere(p.clone(), first_center, 0.4);

    // 2個目以降をSmoothMinで結合していく
    // (Rustのループでグラフを展開する)
    for i in 1..num_spheres {
        let center = centers.clone().slice([i..(i + 1)]).squeeze::<1>(); // [3]
        let dist = sdf_sphere(p.clone(), center, 0.4);
        min_dist = smooth_min(min_dist, dist, 0.2);
    }

    min_dist
}

fn render<B: Backend>(
    ray_org: Tensor<B, 2>,
    ray_dir: Tensor<B, 2>,
    centers: Tensor<B, 2>,
    colors: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_rays = ray_org.dims()[0];
    let num_spheres = centers.dims()[0];
    let device = ray_org.device();
    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    // Ray Marching Loop
    for _ in 0..40 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();

        // シーン全体の距離
        let dist = scene_sdf_value(p, centers.clone());
        t = t + dist;
    }

    let p_final = ray_org + ray_dir * t;

    // 法線 (シーン全体)
    let normal = calc_normal_scene(p_final.clone(), centers.clone());

    // ライティング
    let light_dir = Tensor::<B, 1>::from_floats([-0.5, 0.5, -1.0], &device);
    let light_dir = light_dir.clone() / (light_dir.powf_scalar(2.0).sum().sqrt() + 1e-6);
    let diffuse = (normal * light_dir.unsqueeze()).sum_dim(1).clamp_min(0.0);
    let lighting = diffuse + 0.1;

    // 3. Color Blending (N球混合)
    let mut weight_sum = Tensor::<B, 2>::zeros([num_rays, 1], &device) + 1e-5;
    let mut color_sum = Tensor::<B, 2>::zeros([num_rays, 3], &device);

    for i in 0..num_spheres {
        let center = centers.clone().slice([i..(i + 1)]).squeeze::<1>();
        let color = colors.clone().slice([i..(i + 1)]).squeeze::<1>();
        let dist = sdf_sphere(p_final.clone(), center, 0.4);

        // 重み計算 (指数関数的に柔らかくする)
        let weight = (-dist * 10.0).exp();

        weight_sum = weight_sum + weight.clone();
        color_sum = color_sum + color.unsqueeze() * weight;
    }

    let mixed_color = color_sum / weight_sum;

    let object_color = mixed_color * lighting;

    // マスク
    let dist_scene = scene_sdf_value(p_final, centers);
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
    let target_centers = Tensor::<MyBackend, 1>::from_floats(
        [
            0.2, 0.2, 0.0, // 球1
            -0.2, -0.2, 0.0, 0.0, 0.0, 0.0,
        ], // 球2
        &device,
    )
    .reshape([3, 3]);
    let target_colors = Tensor::<MyBackend, 1>::from_floats(
        [
            1.0, 0.0, 0.0, // 赤
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ], // 青
        &device,
    )
    .reshape([3, 3]);

    // 正解画像作成 (モデルを使わず直接関数を呼ぶ)
    let target_img = render(
        ray_org.clone(),
        ray_dir.clone(),
        target_centers.clone(),
        target_colors.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img.clone(),
        width as u32,
        height as u32,
        "target.png",
    );

    // --- Model Init ---
    // 初期値: 中央付近に(N=3)(shape [3,N])
    let init_centers = Tensor::<MyBackend, 1>::zeros([9], &device).reshape([3, 3]);

    let init_logits = Tensor::<MyBackend, 1>::zeros([9], &device).reshape([3, 3]);

    let mut model = SceneModel::new(init_centers, init_logits);
    let mut optim = AdamConfig::new().init();

    println!("Start Optimization (Metaballs)...");
    let lr = 0.1;

    for i in 0..200 {
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
                model.centers.val(),
                activation::sigmoid(model.colors.val()),
            );
        }
    }

    println!(
        "Final Result:\n  Pos: {}\n  Col: {}",
        model.centers.val(),
        activation::sigmoid(model.colors.val()),
    );
    let img = model.forward(ray_org.clone(), ray_dir.clone());
    save_tensor_as_image(img.clone(), width as u32, height as u32, "final.png");
}
