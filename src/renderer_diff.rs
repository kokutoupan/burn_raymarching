use crate::model::scene::{calc_normal_scene, scene_sdf_value};
use burn::prelude::*;
use burn::tensor::activation;

pub fn render_diff<B: Backend>(
    ray_org: Tensor<B, 2>, // [N, 3]
    ray_dir: Tensor<B, 2>, // [N, 3]
    centers: Tensor<B, 2>, // [M, 3]
    colors: Tensor<B, 2>,  // [M, 3]
    radius: Tensor<B, 2>,  // [M, 1]
    smooth_k: f32,
) -> Tensor<B, 2> // [N, 3]
{
    let num_rays = ray_org.dims()[0];
    let device = ray_org.device();

    let mut t = Tensor::<B, 2>::zeros([num_rays, 1], &device);

    for _ in 0..40 {
        let p = ray_org.clone() + ray_dir.clone() * t.clone();
        let dist = scene_sdf_value(p, centers.clone(), radius.clone(), smooth_k);
        t = (t + dist).detach();
    }

    // 2. 勾配接続フェーズ: 最後の1歩だけ計算グラフに乗せる
    // 見つけた交点付近の座標 (detach済み)
    let p_approx = ray_org.clone() + ray_dir.clone() * t.clone();

    // ★ここがミソ: この SDF 評価には centers と radius の勾配が乗る
    let dist_last = scene_sdf_value(p_approx, centers.clone(), radius.clone(), smooth_k);

    // detach された t に、勾配付きの dist_last を足すことで勾配を「再接続」する
    let t_final = t + dist_last;

    // 最終的な交点 (これで centers と radius を動かすフィードバックが復活する)
    let p_final = ray_org + ray_dir * t_final;

    let normal = calc_normal_scene(
        p_final.clone().detach(),
        centers.clone().detach(),
        radius.clone().detach(),
        smooth_k,
    );

    let light_dir_vec: [f32; 3] = [-0.5, 0.5, -1.0];
    // 正規化
    let ld_len =
        (light_dir_vec[0].powf(2.0) + light_dir_vec[1].powf(2.0) + light_dir_vec[2].powf(2.0))
            .sqrt();
    let light_dir_data = light_dir_vec.map(|x| x / ld_len);

    // [1, 3] として作成 (ブロードキャスト用)
    let light_dir = Tensor::<B, 1>::from_floats(light_dir_data, &device).unsqueeze_dim(0);

    // normal: [N, 3], light_dir: [1, 3] -> [N, 3]
    let diffuse = (normal * light_dir).sum_dim(1).clamp_min(0.0); // [N, 1]

    let lighting = diffuse + 0.1;

    // Step A: 全レイと全球の距離行列 [N, M] を計算
    let p_sq = p_final.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
    let c_sq = centers.clone().powf_scalar(2.0).sum_dim(1).transpose(); // [1, M]
    let p_dot_c = p_final.clone().matmul(centers.clone().transpose()); // [N, M]

    let dists_sq = p_sq + c_sq - p_dot_c * 2.0; // [N, M]
    // 半径を引いて最終的な距離行列を出す
    let dists = dists_sq.clamp_min(1e-6).sqrt() - radius.clone().transpose(); // [N, M]

    // Step B: 重みの計算 [N, M]
    // weight = exp(-dist * 10.0)
    let weights = dists.mul_scalar(-10.0).exp();

    // Step C: 色の加重平均
    // colors: [M, 3] -> [1, M, 3]
    let colors_expanded = colors.unsqueeze_dim::<3>(0);

    // weights: [N, M] -> [N, M, 1] (色成分へ掛けるため)
    let weights_expanded = weights.clone().unsqueeze_dim::<3>(2);

    // weighted_colors: [N, M, 3] = [1, M, 3] * [N, M, 1]
    let weighted_colors = colors_expanded * weights_expanded;

    // 全球分を合計 [N, M, 3] -> [N, 1, 3] -> [N, 3]
    let color_sum = weighted_colors.sum_dim(1).squeeze_dim(1);

    // 重みの合計 [N, M] -> [N, 1]
    let weight_sum = weights.sum_dim(1) + 1e-5;

    // 混合色
    let mixed_color = color_sum / weight_sum;

    // let object_color = mixed_color * lighting;
    let object_color = mixed_color;

    let dist_scene = scene_sdf_value(p_final, centers, radius, smooth_k);

    let mask = activation::sigmoid(dist_scene.mul_scalar(-15.0));

    object_color * mask
}
