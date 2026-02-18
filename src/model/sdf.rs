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
