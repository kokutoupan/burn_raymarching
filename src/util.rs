use burn::prelude::*;
use image::{ColorType, save_buffer};

pub fn save_tensor_as_image<B: Backend>(tensor: Tensor<B, 2>, width: u32, height: u32, path: &str) {
    let floats: Vec<f32> = tensor.into_data().to_vec::<f32>().unwrap();
    let pixels: Vec<u8> = floats
        .iter()
        .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    save_buffer(path, &pixels, width, height, ColorType::Rgb8).expect("Failed to save image");
    println!("Saved image to {}", path);
}
