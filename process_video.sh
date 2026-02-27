#!/bin/bash
set -e # エラーが起きたらそこで止まる

VIDEO_FILE="data/tomato.mp4"
BASE_DIR="data/tomato"
IMG_DIR="$BASE_DIR/images"
DB_PATH="$BASE_DIR/database.db"
SPARSE_DIR="$BASE_DIR/sparse/"

echo "🧹 1. 過去のデータをクリーンアップ..."
rm -rf $BASE_DIR
mkdir -p $IMG_DIR
mkdir -p $SPARSE_DIR

echo "🎞️ 2. ffmpegで動画から画像を多め（10fps）に抽出..."
# 正方形(1024x1024)にクロップ＆パディングして出力
ffmpeg -i $VIDEO_FILE -vf "fps=10,scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:(ow-iw)/2:(oh-ih)/2:black" -q:v 2 $IMG_DIR/img_%04d.jpg

echo "🔍 3. COLMAP: 特徴点抽出..."
colmap feature_extractor \
    --database_path $DB_PATH \
    --image_path $IMG_DIR \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --FeatureExtraction.use_gpu 0

echo "🤝 4. COLMAP: マッチング..."
colmap exhaustive_matcher \
    --database_path $DB_PATH \
    --FeatureMatching.use_gpu 0

echo "🗺️ 5. COLMAP: マッピング（空間構築）..."
# 今回はトマトなので、少しだけ条件を甘くして確実に通します
colmap mapper \
    --database_path $DB_PATH \
    --image_path $IMG_DIR \
    --output_path $SPARSE_DIR \
    --Mapper.min_num_matches 10 \
    --Mapper.init_min_num_inliers 30

echo "📄 6. COLMAP: TXT形式へ変換..."
colmap model_converter \
    --input_path $SPARSE_DIR/0 \
    --output_path $SPARSE_DIR/0 \
    --output_type TXT

echo "🎉 COLMAP処理が完了しました！ sparse/0 の中に cameras.txt と images.txt があります。"