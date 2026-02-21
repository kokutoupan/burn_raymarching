use burn::prelude::*;
use rand::RngExt;

pub struct SceneDataset<B: Backend> {
    pub rays_o: Tensor<B, 2>,
    pub rays_d: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
    pub num_total_pixels: usize,
    pub fg_indices: Vec<i32>,
    pub bg_indices: Vec<i32>,
}

impl<B: Backend> SceneDataset<B> {
    pub fn new(rays_o: Tensor<B, 2>, rays_d: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        let num_total_pixels = rays_o.dims()[0];
        let targets_data: Vec<f32> = targets
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut fg_indices = Vec::new();
        let mut bg_indices = Vec::new();

        for i in 0..num_total_pixels {
            let idx = i * 3;
            let sum_color = targets_data[idx] + targets_data[idx + 1] + targets_data[idx + 2];

            if sum_color > 0.05 {
                fg_indices.push(i as i32);
            } else {
                bg_indices.push(i as i32);
            }
        }

        Self {
            rays_o,
            rays_d,
            targets,
            num_total_pixels,
            fg_indices,
            bg_indices,
        }
    }

    pub fn sample_batch(
        &self,
        batch_size: usize,
        uniform_ratio: f32,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let mut rng = rand::rng();
        let mut uniform_batch_size = (batch_size as f32 * uniform_ratio) as usize;
        let mut fg_boost_batch_size = batch_size - uniform_batch_size;

        if !self.fg_indices.is_empty() && self.fg_indices.len() < fg_boost_batch_size {
            fg_boost_batch_size = self.fg_indices.len();
            uniform_batch_size = batch_size - fg_boost_batch_size;
        }

        let mut batch_indices = Vec::with_capacity(batch_size);

        for _ in 0..uniform_batch_size {
            batch_indices.push(rng.random_range(0..self.num_total_pixels as i32));
        }

        if !self.fg_indices.is_empty() && fg_boost_batch_size > 0 {
            for _ in 0..fg_boost_batch_size {
                let idx = rng.random_range(0..self.fg_indices.len());
                batch_indices.push(self.fg_indices[idx]);
            }
        }

        let indices = Tensor::<B, 1, Int>::from_ints(batch_indices.as_slice(), device);

        let batch_ro = self.rays_o.clone().select(0, indices.clone()).detach();
        let batch_rd = self.rays_d.clone().select(0, indices.clone()).detach();
        let batch_target = self.targets.clone().select(0, indices).detach();

        (batch_ro, batch_rd, batch_target)
    }
}
