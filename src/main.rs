use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation;
use image::{ColorType, save_buffer};

// --- 0. 依存なしの簡易算術ヘルパー ---
mod math {
    pub type Vec3 = [f32; 3];

    pub fn normalize(v: Vec3) -> Vec3 {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len == 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [v[0] / len, v[1] / len, v[2] / len]
        }
    }

    pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
}

// --- 1. 完全なカメラレイ生成関数 (LookAt対応) ---
fn create_camera_rays<B: Backend>(
    width: usize,
    height: usize,
    eye: [f32; 3],    // カメラの位置
    target: [f32; 3], // カメラが見る点
    fov_deg: f32,     // 画角 (例: 60.0)
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let num_rays = width * height;

    // 1. カメラ座標系の基底ベクトルを計算
    let world_up = [0.0, 1.0, 0.0];
    let forward = math::normalize(math::sub(target, eye)); // Z+ (Look Dir)
    let right = math::normalize(math::cross(forward, world_up)); // X+
    let up = math::cross(right, forward); // Y+

    // 2. 画角から焦点距離などを計算
    // aspect ratio
    let aspect = width as f32 / height as f32;
    // FOVから画面の大きさを逆算 (tan(theta/2))
    let theta = fov_deg.to_radians() / 2.0;
    let half_height = theta.tan();
    let half_width = aspect * half_height;

    // 3. 全ピクセルのレイ方向を計算

    let mut dirs = Vec::with_capacity(num_rays * 3);

    for y in 0..height {
        for x in 0..width {
            // UV座標 (-1.0 ~ 1.0)
            // Y軸は上がプラスになるように反転させるのが一般的
            let u = (x as f32 / width as f32) * 2.0 - 1.0;
            let v = -((y as f32 / height as f32) * 2.0 - 1.0);

            // スクリーン上の点へのベクトル (Camera Space)
            // ray = u * Right * half_width + v * Up * half_height + Forward
            let r_scale = u * half_width;
            let u_scale = v * half_height;

            let dx = right[0] * r_scale + up[0] * u_scale + forward[0];
            let dy = right[1] * r_scale + up[1] * u_scale + forward[1];
            let dz = right[2] * r_scale + up[2] * u_scale + forward[2];

            // Normalize
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            dirs.push(dx / len);
            dirs.push(dy / len);
            dirs.push(dz / len);
        }
    }

    // Tensor作成
    let ray_org = Tensor::<B, 1>::from_floats(eye, device)
        .reshape([1, 3])
        .repeat_dim(0, num_rays); // [N, 3]

    let ray_dir = Tensor::<B, 1>::from_floats(dirs.as_slice(), device).reshape([num_rays, 3]); // [N, 3]

    (ray_org, ray_dir)
}

// --- 1. モデル定義 (Module) ---
#[derive(Module, Debug)]
struct SceneModel<B: Backend> {
    centers: Param<Tensor<B, 2>>, // [N, 3]
    colors: Param<Tensor<B, 2>>,  // [N, 3]
    radius: Param<Tensor<B, 2>>,  // [N, 1] 半径 (スカラだが扱いやすくするため2次元配列)
}

impl<B: Backend> SceneModel<B> {
    fn new(centers: Tensor<B, 2>, colors: Tensor<B, 2>, radius: Tensor<B, 2>) -> Self {
        Self {
            centers: Param::from_tensor(centers),
            colors: Param::from_tensor(colors),
            radius: Param::from_tensor(radius),
        }
    }

    fn forward(&self, ray_org: Tensor<B, 2>, ray_dir: Tensor<B, 2>) -> Tensor<B, 2> {
        let colors_rgb = activation::sigmoid(self.colors.val());
        let centers = self.centers.val();
        let radius_positive = activation::softplus(self.radius.val(), 1.0) + 0.01;

        render(ray_org, ray_dir, centers, colors_rgb, radius_positive)
    }
}

// SDF
fn sdf_sphere<B: Backend>(
    p: Tensor<B, 2>,
    center: Tensor<B, 1>,
    radius: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let diff = p - center.unsqueeze();
    (diff.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt() - radius.unsqueeze()
}

// k: 溶け具合 (0.1〜0.5くらい)
fn smooth_min<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, k: f32) -> Tensor<B, 2> {
    // h = max(k - |a - b|, 0) / k
    let h = (a.clone() - b.clone())
        .abs()
        .neg()
        .add_scalar(k)
        .clamp_min(0.0)
        .div_scalar(k);

    // result = min(a, b) - h^2 * k / 4
    let min_ab = a.min_pair(b);
    min_ab - h.powf_scalar(2.0) * (k * 0.25)
}

fn calc_normal_scene<B: Backend>(
    p: Tensor<B, 2>,
    centers: Tensor<B, 2>,
    radius: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let eps = 1e-4;
    let e_x = Tensor::<B, 1>::from_floats([eps, 0.0, 0.0], &p.device()).unsqueeze();
    let e_y = Tensor::<B, 1>::from_floats([0.0, eps, 0.0], &p.device()).unsqueeze();
    let e_z = Tensor::<B, 1>::from_floats([0.0, 0.0, eps], &p.device()).unsqueeze();

    // シーン全体のSDFを計算するクロージャ的ヘルパー
    let get_dist = |pos: Tensor<B, 2>| -> Tensor<B, 2> {
        scene_sdf_value(pos, centers.clone(), radius.clone())
    };

    let nx = get_dist(p.clone() + e_x.clone()) - get_dist(p.clone() - e_x);
    let ny = get_dist(p.clone() + e_y.clone()) - get_dist(p.clone() - e_y);
    let nz = get_dist(p.clone() + e_z.clone()) - get_dist(p.clone() - e_z);

    let n = Tensor::cat(vec![nx, ny, nz], 1);
    let len = (n.clone().powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    n / len
}

// 配列からSDF値を計算する関数 (ループ処理)
fn scene_sdf_value<B: Backend>(
    p: Tensor<B, 2>,
    centers: Tensor<B, 2>,
    radius: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_spheres = centers.dims()[0];

    // 最初の球の距離で初期化
    let first_center = centers.clone().slice([0..1]).reshape([3]);
    let mut min_dist = sdf_sphere(
        p.clone(),
        first_center,
        radius.clone().slice([0..1]).reshape([1]),
    );

    // 2個目以降をSmoothMinで結合していく
    // (Rustのループでグラフを展開する)
    for i in 1..num_spheres {
        let center = centers.clone().slice([i..(i + 1)]).reshape([3]); // [3]
        let radius = radius.clone().slice([i..(i + 1)]).reshape([1]); // [1]
        let dist = sdf_sphere(p.clone(), center, radius);
        min_dist = smooth_min(min_dist, dist, 0.2);
    }

    min_dist
}

fn render<B: Backend>(
    ray_org: Tensor<B, 2>,
    ray_dir: Tensor<B, 2>,
    centers: Tensor<B, 2>,
    colors: Tensor<B, 2>,
    radius: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_rays = ray_org.dims()[0];
    let num_spheres = centers.dims()[0];
    let device = ray_org.device();
    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    // Ray Marching Loop
    for _ in 0..40 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();

        // シーン全体の距離
        let dist = scene_sdf_value(p, centers.clone(), radius.clone());
        t = t + dist;
    }

    let p_final = ray_org + ray_dir * t;

    // 法線 (シーン全体)
    let normal = calc_normal_scene(p_final.clone(), centers.clone(), radius.clone());

    // ライティング
    let light_dir = Tensor::<B, 1>::from_floats([-0.5, 0.5, -1.0], &device);
    let light_dir = light_dir.clone() / (light_dir.powf_scalar(2.0).sum().sqrt() + 1e-6);
    let diffuse = (normal * light_dir.unsqueeze()).sum_dim(1).clamp_min(0.0);
    let lighting = diffuse + 0.1;
    // let lighting = Tensor::<B, 2>::ones([num_rays, 1], &device);

    // 3. Color Blending (N球混合)
    let mut weight_sum = Tensor::<B, 2>::zeros([num_rays, 1], &device) + 1e-5;
    let mut color_sum = Tensor::<B, 2>::zeros([num_rays, 3], &device);

    for i in 0..num_spheres {
        let center = centers.clone().slice([i..(i + 1)]).reshape([3]);
        let color = colors.clone().slice([i..(i + 1)]).reshape([3]);
        let r = radius.clone().slice([i..(i + 1)]).reshape([1]);
        let dist = sdf_sphere(p_final.clone(), center, r);

        // 重み計算 (指数関数的に柔らかくする)
        let weight = (-dist * 10.0).exp();

        weight_sum = weight_sum + weight.clone();
        color_sum = color_sum + color.unsqueeze() * weight;
    }

    let mixed_color = color_sum / weight_sum;

    let object_color = mixed_color * lighting;

    // マスク
    let dist_scene = scene_sdf_value(p_final, centers, radius);
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

    // --- 1. カメラ設定 (3視点) ---
    // Camera 1: 正面
    let (ray_org_1, ray_dir_1) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 0.0, -2.5], // Eye: 上手前
        [0.0, 0.0, 0.0],  // Target: 中心
        50.0,             // FOV: 50度
        &device,
    );

    // Camera 2: 真横 (右から)
    let (ray_org_2, ray_dir_2) = create_camera_rays::<MyBackend>(
        width,
        height,
        [2.5, 0.0, 0.0], // Eye: 右
        [0.0, 0.0, 0.0], // Target: 中心
        50.0,            // FOV
        &device,
    );

    // Camera 3: 真上(少しずらす)
    let (ray_org_3, ray_dir_3) = create_camera_rays::<MyBackend>(
        width,
        height,
        [0.0, 2.5, -0.001], // Eye: 上
        [0.0, 0.0, 0.0],    // Target: 中心
        50.0,               // FOV
        &device,
    );

    // --- Target: 3つの球 (Radius違い) ---
    // 左(赤,大), 中(緑,小), 右(青,中)
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

    let target_radii = Tensor::<MyBackend, 1>::from_floats(
        [0.2, 0.15, 0.2], // 大きさもバラバラにしてみる
        &device,
    )
    .reshape([3, 1]); // [3, 1]

    // 正解画像を2枚作る (Front & Side)
    let target_img_1 = render(
        ray_org_1.clone(),
        ray_dir_1.clone(),
        target_centers.clone(),
        target_colors.clone(),
        target_radii.clone(),
    )
    .detach();
    let target_img_2 = render(
        ray_org_2.clone(),
        ray_dir_2.clone(),
        target_centers.clone(),
        target_colors.clone(),
        target_radii.clone(),
    )
    .detach();
    save_tensor_as_image(
        target_img_1.clone(),
        width as u32,
        height as u32,
        "target_1.png",
    );
    save_tensor_as_image(
        target_img_2.clone(),
        width as u32,
        height as u32,
        "target_2.png",
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
        target_img_3.clone(),
        width as u32,
        height as u32,
        "target_3.png",
    );

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
                &format!("step_{}.png", i),
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
        "final_1.png",
    );
    save_tensor_as_image(
        final_img2.clone(),
        width as u32,
        height as u32,
        "final_2.png",
    );
    save_tensor_as_image(
        final_img3.clone(),
        width as u32,
        height as u32,
        "final_3.png",
    );
}
