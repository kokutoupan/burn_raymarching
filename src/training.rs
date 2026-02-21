use burn::prelude::*;
use burn::tensor::activation;
use rand::RngExt;

use crate::model::scene::SceneModel;

pub fn compute_loss<B: Backend>(
    model: &SceneModel<B>,
    output: Tensor<B, 2>,
    batch_target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let device = output.device();

    // 1. 画像再構成Loss (メインの学習目標)
    let diff = output - batch_target.clone();
    let mse_map = diff.clone().powf_scalar(2.0);
    let abs_diff = diff.abs();

    let target_sum = batch_target.sum_dim(1);
    let target_mask = target_sum.greater_elem(0.01); // 物体領域の判定

    // 物体領域は厳しく(L1 * 10)、背景は緩く(MSE)
    let reconstruction_loss = mse_map.mask_where(target_mask, abs_diff * 10.0).mean();
    let mut loss = reconstruction_loss;

    // 2. 幾何学的制約 (ペナルティ項)
    let current_n = model.centers.dims()[0];
    let centers = model.centers.val();
    let radii = activation::softplus(model.radius.val(), 1.0);

    // [a] 半径ペナルティ: 球が大きくなりすぎるのを防ぎつつ、不要な球を小さくする
    let radius_l1_penalty = radii.clone().abs().mean();

    let r_mask = radii.clone().greater_elem(1.0);
    let radius_large_penalty = Tensor::zeros_like(&radii)
        .mask_where(r_mask, radii.clone().powf_scalar(2.0))
        .mean();

    loss = loss + radius_large_penalty * 0.1 + radius_l1_penalty * 0.004;

    // [b] 原点引力: 球がバラバラに散らばるのを防ぐ
    let center_penalty = centers.clone().powf_scalar(2.0).mean();
    loss = loss + center_penalty * 0.1; // Billboard対策で少し強めに設定

    // [c] カメラ近接バリア (Billboard Effect対策の要)
    let centers_val = model.centers.val();
    let dist_from_origin = (centers_val.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    let max_reach = dist_from_origin + radii;

    let out_of_bounds_mask = max_reach.clone().greater_elem(1.2);
    let excess_dist = max_reach.clone() - 1.1;
    let penalty_values = excess_dist.powf_scalar(2.0);

    let camera_proximity_penalty = Tensor::zeros_like(&max_reach)
        .mask_where(out_of_bounds_mask, penalty_values)
        .mean();

    loss = loss + camera_proximity_penalty * 5.0;

    // [d] 反発項: 球同士の重なりを防ぐ
    let centers_val = model.centers.val();
    let c_sq_val = centers_val.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
    let c_sq_t = c_sq_val.clone().transpose(); // [1, N]
    let c_dot_c = centers_val.clone().matmul(centers_val.clone().transpose()); // [N, N]

    let dist_sq = c_sq_val + c_sq_t - c_dot_c * 2.0; // [N, N]
    let dist_matrix = dist_sq.clamp_min(1e-6).sqrt(); // [N, N]
    let eye = Tensor::<B, 2>::eye(current_n, &device);
    let repulsion_loss = (dist_matrix + eye * 100.0 + 1e-6).powf_scalar(-1.0).mean();
    loss = loss + repulsion_loss * 0.00001;

    loss
}

pub fn prune_and_split<B: Backend>(
    model: &SceneModel<B>,
    stage: usize,
    stages: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let current_n = model.centers.dims()[0];
    let out_centers: Vec<f32> = model
        .centers
        .val()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let out_colors: Vec<f32> = model
        .colors
        .val()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let eval_radii: Vec<f32> = activation::softplus(model.radius.val(), 1.0)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let mut next_centers = Vec::new();
    let mut next_colors = Vec::new();
    let mut next_radii = Vec::new();
    let mut mut_rng = rand::rng();

    for i in 0..current_n {
        let r = eval_radii[i];
        let cx = out_centers[i * 3];
        let cy = out_centers[i * 3 + 1];
        let cz = out_centers[i * 3 + 2];
        let cr = out_colors[i * 3];
        let cg = out_colors[i * 3 + 1];
        let cb = out_colors[i * 3 + 2];

        // 1. Pruning (削除): デカすぎる影(0.25超え)や、極小のゴミ(0.01未満)は次世代に引き継がない
        if r > 0.25 || r < 0.01 {
            continue;
        }

        // 2. Keep (維持)
        next_centers.extend_from_slice(&[cx, cy, cz]);
        next_colors.extend_from_slice(&[cr, cg, cb]);
        next_radii.push(-2.5);

        // 3. Splitting (分裂)
        if stage < stages - 1 {
            let offset_x = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;
            let offset_y = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;
            let offset_z = (mut_rng.random_range(0.0..1.0) - 0.5) * 0.05;

            next_centers.extend_from_slice(&[cx + offset_x, cy + offset_y, cz + offset_z]);
            next_colors.extend_from_slice(&[cr, cg, cb]);
            next_radii.push(-2.5);
        }
    }

    let next_n = next_radii.len();
    (next_centers, next_colors, next_radii, next_n)
}
