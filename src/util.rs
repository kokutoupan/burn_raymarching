use burn::prelude::*;
use image::{ColorType, GenericImageView, save_buffer};

pub fn save_tensor_as_image<B: Backend>(tensor: Tensor<B, 2>, width: u32, height: u32, path: &str) {
    let floats: Vec<f32> = tensor.into_data().to_vec::<f32>().unwrap();
    let pixels: Vec<u8> = floats
        .iter()
        .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create directory");
        }
    }

    save_buffer(path, &pixels, width, height, ColorType::Rgb8).expect("Failed to save image");
    println!("Saved image to {}", path);
}

pub fn load_image_as_tensor<B: Backend>(path: &str, device: &B::Device) -> Tensor<B, 2> {
    let img = image::open(path).expect("Failed to open image");
    let (width, height) = img.dimensions();
    let pixels = img.to_rgb8().into_raw();

    let floats: Vec<f32> = pixels.iter().map(|&x| (x as f32) / 255.0).collect();

    // [H*W, 3] -> [NumRays, 3]
    Tensor::<B, 1>::from_floats(floats.as_slice(), device).reshape([(width * height) as usize, 3])
}
