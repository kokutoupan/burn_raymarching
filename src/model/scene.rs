use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation;

use crate::model::sdf::soft_min_tensor;
use crate::renderer_diff::render_diff;

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

        render_diff(ray_org, ray_dir, centers, colors_rgb, radius_positive)
    }
}

// 配列からSDF値を計算する関数 (ループ処理)
pub fn scene_sdf_value<B: Backend>(
    p: Tensor<B, 2>,       // [N, 3]
    centers: Tensor<B, 2>, // [M, 3]
    radius: Tensor<B, 2>,  // [M, 1]
) -> Tensor<B, 2> {
    let num_spheres = centers.dims()[0];
    let num_points = p.dims()[0];

    let p_expanded = p.unsqueeze_dim::<3>(1);
    let centers_expanded = centers.unsqueeze_dim::<3>(0);
    // [N, 1, 3] - [1, M, 3] = [N, M, 3]
    let diff = p_expanded - centers_expanded;
    let dists = diff
        .powf_scalar(2.0)
        .sum_dim(2) // [N, M, 1]
        .sqrt()
        .squeeze_dim(2); // [N, M]

    // 半径を引く (radius: [M, 1] -> [1, M] にして引く)
    let radius_row = radius.transpose(); // [1, M]
    let all_dists = dists - radius_row; // [N, M] - [1, M] -> [N, M]

    // let mut final_dist = all_dists.clone().slice([0..num_points, 0..1]);
    // for i in 1..num_spheres {
    //     // 次の球の距離列を取得 [N, 1]
    //     let next_dist = all_dists.clone().slice([0..num_points, i..(i + 1)]);

    //     // 結合
    //     final_dist = smooth_min(final_dist, next_dist, 0.2);
    // }

    soft_min_tensor(all_dists, 32.0)
}

pub fn calc_normal_scene<B: Backend>(
    p: Tensor<B, 2>,       // [N, 3]
    centers: Tensor<B, 2>, // [M, 3]
    radius: Tensor<B, 2>,  // [M, 1]
) -> Tensor<B, 2> {
    // [N, 3] (Normal)

    let n_points = p.dims()[0];
    let device = p.device();
    let eps = 1e-4;

    // 1. オフセット行列を作成 [6, 3]
    // (+x, -x, +y, -y, +z, -z) の順
    let offsets_data = [
        eps, 0.0, 0.0, -eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, -eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, -eps,
    ];
    let offsets = Tensor::<B, 1>::from_floats(offsets_data, &device).reshape([6, 3]);

    // 2. pを拡張してオフセットを足す
    // p: [N, 3] -> [N, 1, 3]
    // offsets: [6, 3] -> [1, 6, 3]
    // p_expanded: [N, 6, 3] (ブロードキャスト加算)
    let p_expanded = p.unsqueeze_dim::<3>(1) + offsets.unsqueeze_dim::<3>(0);

    // 3. バッチ処理のためにフラット化 [N*6, 3]
    let p_flat = p_expanded.reshape([n_points * 6, 3]);

    // 4. 一括でSDF計算
    // 結果は [N*6, 1]
    let dists_flat = scene_sdf_value(p_flat, centers, radius);

    // 5. 結果を [N, 6] に戻す
    // 列0: +x, 列1: -x, 列2: +y... と並んでいます
    let dists = dists_flat.reshape([n_points, 6]);

    // 6. 差分計算 (Central Difference)
    // distsの各列を取り出して計算
    let dx = dists.clone().slice([0..n_points, 0..1]) - dists.clone().slice([0..n_points, 1..2]);
    let dy = dists.clone().slice([0..n_points, 2..3]) - dists.clone().slice([0..n_points, 3..4]);
    let dz = dists.clone().slice([0..n_points, 4..5]) - dists.clone().slice([0..n_points, 5..6]);

    // 7. 法線ベクトルの結合と正規化
    let normal = Tensor::cat(vec![dx, dy, dz], 1); // [N, 3]
    let len = (normal.clone().powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();

    normal / len
}
