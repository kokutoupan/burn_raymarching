use image::{GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use std::fs;

fn main() {
    let input_dir = "data/tomato/images";
    let output_dir = "data/tomato/images_nobg";
    fs::create_dir_all(output_dir).unwrap();

    for entry in fs::read_dir(input_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().unwrap_or_default() != "jpg" {
            continue;
        }

        // 1. 画像を読み込む
        let img = image::open(&path).unwrap();

        // 2. ピクセル処理の前に 256x256 にリサイズ (Lanczos3フィルターが高画質でおすすめです)
        let resized_img = img.resize_exact(256, 256, FilterType::Lanczos3);

        let (width, height) = resized_img.dimensions();
        // RGBのキャンバスを作成 (256x256)
        let mut out_img = ImageBuffer::new(width, height);

        for (x, y, pixel) in resized_img.pixels() {
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            let rf = r as f32;
            let gf = g as f32;
            let bf = b as f32;

            let brightness = rf * rf + gf * gf + bf * bf;

            //はしは消す
            if brightness > 150.0 * 150.0
                && rf > bf + 20.0
                && x > 50
                && x < 206
                && y > 50
                && y < 206
            {
                out_img.put_pixel(x, y, Rgb([r, g, b]));
            } else {
                // 背景は完全な黒に
                out_img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }

        let out_path = format!(
            "{}/{}",
            output_dir,
            path.file_name().unwrap().to_str().unwrap()
        );
        out_img.save(out_path).unwrap();
    }
    println!("✅ 背景抜き＆256x256リサイズ完了！ data/banana/images_nobg を確認してください！");
}
