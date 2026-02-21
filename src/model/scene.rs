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
    pub light_dir: Param<Tensor<B, 1>>,
    pub ambient_intensity: Param<Tensor<B, 1>>,
}

impl<B: Backend> SceneModel<B> {
    pub fn new(
        centers: Tensor<B, 2>,
        colors: Tensor<B, 2>,
        radius: Tensor<B, 2>,
        light_dir: Tensor<B, 1>,
        ambient_intensity: Tensor<B, 1>,
    ) -> Self {
        Self {
            centers: Param::from_tensor(centers),
            colors: Param::from_tensor(colors),
            radius: Param::from_tensor(radius),
            light_dir: Param::from_tensor(light_dir),
            ambient_intensity: Param::from_tensor(ambient_intensity),
        }
    }

    pub fn forward(
        &self,
        ray_org: Tensor<B, 2>,
        ray_dir: Tensor<B, 2>,
        smooth_k: f32,
    ) -> Tensor<B, 2> {
        let colors_rgb = activation::sigmoid(self.colors.val());
        let centers = self.centers.val();
        let radius_positive = activation::softplus(self.radius.val(), 1.0) + 0.01;
        let light_dir = self.light_dir.val();
        let ambient_intensity = self.ambient_intensity.val();

        render_diff(
            ray_org,
            ray_dir,
            centers,
            colors_rgb,
            radius_positive,
            light_dir,
            ambient_intensity,
            smooth_k,
        )
    }
}

pub fn scene_sdf_value<B: Backend>(
    p: Tensor<B, 2>,       // [N, 3]
    centers: Tensor<B, 2>, // [M, 3]
    radius: Tensor<B, 2>,  // [M, 1]
    smooth_k: f32,
) -> Tensor<B, 2> {
    // 展開公式 (||p - c||^2 = ||p||^2 + ||c||^2 - 2p*c) を使って距離を計算
    let p_sq = p.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
    let c_sq = centers.clone().powf_scalar(2.0).sum_dim(1).transpose(); // [1, M]
    let p_dot_c = p.matmul(centers.transpose()); // [N, M]

    let dists_sq = p_sq + c_sq - p_dot_c * 2.0; // [N, M]
    let dists = dists_sq.clamp_min(1e-6).sqrt(); // [N, M]

    // 半径を引く (radius: [M, 1] -> [1, M] にして引く)
    let radius_row = radius.transpose(); // [1, M]
    let all_dists = dists - radius_row; // [N, M] - [1, M] -> [N, M]

    soft_min_tensor(all_dists, smooth_k)
}

pub fn calc_normal_scene<B: Backend>(
    p: Tensor<B, 2>,       // [N, 3]
    centers: Tensor<B, 2>, // [M, 3]
    radius: Tensor<B, 2>,  // [M, 1]
    smooth_k: f32,
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
    let dists_flat = scene_sdf_value(p_flat, centers, radius, smooth_k);

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
