use burn::prelude::*;

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
pub fn create_camera_rays<B: Backend>(
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
