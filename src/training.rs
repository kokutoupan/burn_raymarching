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

    loss = loss + radius_large_penalty * 0.04 + radius_l1_penalty * 0.002;

    // [b] 原点引力: 球がバラバラに散らばるのを防ぐ
    let center_penalty = centers.clone().powf_scalar(2.0).mean();
    loss = loss + center_penalty * 0.05; // Billboard対策で少し強めに設定

    // [c] カメラ近接バリア (Billboard Effect対策の要)
    let centers_val = model.centers.val();
    let dist_from_origin = (centers_val.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt();
    let max_reach = dist_from_origin + radii;

    let out_of_bounds_mask = max_reach.clone().greater_elem(1.2);
    let excess_dist = max_reach.clone() - 1.2;
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
    init_centers: &[f32],
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
    // 次世代に引き継ぐための「生の値 (Logit)」
    let raw_colors: Vec<f32> = model
        .colors
        .val()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    // 削除判定をするための「評価後の値 (0.0 ~ 1.0)」
    let eval_colors: Vec<f32> = activation::sigmoid(model.colors.val())
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let raw_radii: Vec<f32> = model
        .radius
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
        let raw_radius = raw_radii[i];
        let cx = out_centers[i * 3];
        let cy = out_centers[i * 3 + 1];
        let cz = out_centers[i * 3 + 2];

        // 初期座標を取得
        let init_cx = init_centers[i * 3];
        let init_cy = init_centers[i * 3 + 1];
        let init_cz = init_centers[i * 3 + 2];

        // 移動距離の2乗を計算
        let dx = cx - init_cx;
        let dy = cy - init_cy;
        let dz = cz - init_cz;
        let move_dist_sq = dx * dx + dy * dy + dz * dz;

        // 引き継ぎ用の生の色
        let raw_r = raw_colors[i * 3];
        let raw_g = raw_colors[i * 3 + 1];
        let raw_b = raw_colors[i * 3 + 2];

        // 判定用の0.0~1.0の色
        let eval_r = eval_colors[i * 3];
        let eval_g = eval_colors[i * 3 + 1];
        let eval_b = eval_colors[i * 3 + 2];

        // ==========================================
        // 1. 厳密な Pruning (ゴミの徹底排除)
        // ==========================================
        // [a] サイズ異常（デカすぎる影、または小さすぎるゴミ）
        if r > 0.35 || r < 0.005 {
            continue;
        }

        // [b] 画面外への飛散（原点から遠すぎる球は削除: 1.2^2 = 1.44）
        let dist_sq = cx * cx + cy * cy + cz * cz;
        if dist_sq > 1.44 {
            continue;
        }

        // [c] 真っ黒な球（色がない＝マスクや辻褄合わせに使われている不要な球）
        if eval_r + eval_g + eval_b < 0.05 {
            continue;
        }

        // ==========================================
        // 2. 条件付き Splitting (必要なものだけ割る)
        // ==========================================
        if stage < stages - 1 {
            if r > 0.05 && move_dist_sq < 0.05 * 0.05 {
                // 【分割】半径が大きい ＝ まだ大まかな領域しかカバーできていない
                // -> 2つの小さな球に割って、ディテールを表現させる
                let offset = 0.03; // 少しずらす
                let new_r = raw_radius - 0.5;

                // 1つ目
                next_centers.extend_from_slice(&[cx + offset, cy, cz + offset]);
                next_colors.extend_from_slice(&[raw_r, raw_g, raw_b]);
                next_radii.push(new_r); // 半径を小さくリセット(softplusで約0.08)

                // 2つ目（逆方向にずらす）
                next_centers.extend_from_slice(&[cx - offset, cy, cz - offset]);
                next_colors.extend_from_slice(&[raw_r, raw_g, raw_b]);
                next_radii.push(new_r);
            } else {
                // 【維持】すでに十分小さい ＝ ディテールとして綺麗にフィットしている
                // -> 無駄に割らずに、そのまま維持する
                next_centers.extend_from_slice(&[cx, cy, cz]);
                next_colors.extend_from_slice(&[raw_r, raw_g, raw_b]);
                next_radii.push(raw_radius);
            }
        } else {
            next_centers.extend_from_slice(&[cx, cy, cz]);
            next_colors.extend_from_slice(&[raw_r, raw_g, raw_b]);
            next_radii.push(raw_radius);
        }
    }

    let next_n = next_radii.len();
    (next_centers, next_colors, next_radii, next_n)
}
