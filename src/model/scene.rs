use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation;

use crate::model::sdf::{sdf_sphere, smooth_min};
use crate::renderer::render;

// --- 1. モデル定義 (Module) ---
#[derive(Module, Debug)]
pub struct SceneModel<B: Backend> {
    pub centers: Param<Tensor<B, 2>>, // [N, 3]
    pub colors: Param<Tensor<B, 2>>,  // [N, 3]
    pub radius: Param<Tensor<B, 2>>,  // [N, 1] 半径 (スカラだが扱いやすくするため2次元配列)
}

impl<B: Backend> SceneModel<B> {
    pub fn new(centers: Tensor<B, 2>, colors: Tensor<B, 2>, radius: Tensor<B, 2>) -> Self {
        Self {
            centers: Param::from_tensor(centers),
            colors: Param::from_tensor(colors),
            radius: Param::from_tensor(radius),
        }
    }

    pub fn forward(&self, ray_org: Tensor<B, 2>, ray_dir: Tensor<B, 2>) -> Tensor<B, 2> {
        let colors_rgb = activation::sigmoid(self.colors.val());
        let centers = self.centers.val();
        let radius_positive = activation::softplus(self.radius.val(), 1.0) + 0.01;

        render(ray_org, ray_dir, centers, colors_rgb, radius_positive)
    }
}

// 配列からSDF値を計算する関数 (ループ処理)
pub fn scene_sdf_value<B: Backend>(
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

pub fn calc_normal_scene<B: Backend>(
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
