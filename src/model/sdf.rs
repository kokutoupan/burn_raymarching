use burn::prelude::*;

// SDF: Sphere
pub fn sdf_sphere<B: Backend>(
    p: Tensor<B, 2>,
    center: Tensor<B, 1>,
    radius: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let diff = p - center.unsqueeze();
    (diff.powf_scalar(2.0).sum_dim(1) + 1e-6).sqrt() - radius.unsqueeze()
}

// k: 溶け具合 (0.1〜0.5くらい)
pub fn smooth_min<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, k: f32) -> Tensor<B, 2> {
    // h = max(k - |a - b|, 0) / k
    let h = (a.clone() - b.clone())
        .abs()
        .neg()
        .add_scalar(k)
        .clamp_min(0.0)
        .div_scalar(k);

    // result = min(a, b) - h^2 * k / 4
    let min_ab = a.min_pair(b);
    min_ab - h.powf_scalar(2.0).mul_scalar(k * 0.25)
}

// 安定版 LogSumExp (SoftMin)
// k: 滑らかさ係数 (大きいほど鋭い min になる。32.0くらいが目安)
pub fn soft_min_tensor<B: Backend>(dists: Tensor<B, 2>, k: f32) -> Tensor<B, 2> {
    // dists: [N, M] (N: Rays, M: Spheres)
    // Formula: -log( sum( exp(-k * d) ) ) / k

    // 数値安定性のため、最大値を引いてからexp計算する (LogSumExp trick)
    // -k * d なので、最大値は -k * d の中での最大値
    let val = dists.mul_scalar(-k);
    let max_val = val.clone().detach().max_dim(1);

    let exp = (val - max_val.clone()).exp();
    let sum_exp = exp.sum_dim(1); // [N, 1]

    // sum_exp が 0 になると log(0) = -inf となり NaN を引き起こすため、小さな値を足して防ぐ
    (sum_exp.clamp_min(1e-8).log() + max_val).div_scalar(-k)
}
