use crate::model::scene::{calc_normal_scene, scene_sdf_value};
use crate::model::sdf::sdf_sphere;
use burn::prelude::*;

pub fn render<B: Backend>(
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
