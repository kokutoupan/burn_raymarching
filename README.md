# 🍡 Burn Raymarching: 3D Gaussian Splatting-like Differentiable Renderer in Rust

**Burn Raymarching** は、Rustのディープラーニングフレームワーク「[Burn](https://www.google.com/search?q=https://github.com/tracemachina/burn)」を用いて構築された、微分可能レイマーチングエンジンです。

複数の2D画像から、最適化アルゴリズム（Adam）のみを用いて、3D空間上の球体群（位置、色、半径）および光源パラメータを逆算し、3Dシーンを再構築します。近年注目される **3D Gaussian Splatting (3DGS)** の適応的密度制御の概念を、SDF（符号付き距離場）とレイマーチングベースのレンダリングパイプラインに統合した実験的な実装です。

## ✨ 主な特徴 (Key Features)

* **微分可能レイマーチング (Differentiable Raymarching)**
    * SDFベースの球体レンダリングをBurnのTensor演算で実装。
    * 深度合成における不連続性を排除するため、Softmax関数を用いた加重平均ベースのカラーブレンディングを採用し、安定した勾配伝播を実現。


* **適応的密度制御 (Adaptive Pruning & Splitting)**
    * オプティマイザの更新過程において、動的にプリミティブ（球体）の数を増減させるアルゴリズムを実装。
* **Pruning:** 一定の物理半径を超えるもの、または透明度（色の閾値）が基準に満たないプリミティブを定期的に削除。
* **Splitting:** 世代間での「座標の移動距離」と「半径」を評価関数とし、誤差地形において収束していない（移動量が大きい）プリミティブを分割して局所的な表現解像度を向上。


* **光源と環境光の同時最適化 (Illumination Optimization)**
    * 平行光源ベクトル（Light Direction）および環境光強度（Ambient Intensity）を学習パラメータに含め、ジオメトリ（形状）・カラー（反射率）・ライティングの分離（Disentanglement）を実現。


* **多視点学習 (Multi-view Training)**
    * json形式（`cameras.json`）による多視点カメラポーズの動的読み込みに対応。
    * 広範な視点からのLossを計算することで、インバースレンダリング特有の局所解（Billboarding Effect等）を抑制。


* **リアルタイムViewer (Real-time Viewer)**
    * 学習結果（JSON形式）を読み込み、Rust/WGPUおよびWGSLコンピュートシェーダーを用いて推論結果をリアルタイムに描画。



## 🚀 使い方 (How to Run)

### 1. データセットの生成

学習用のターゲット画像群を多視点から生成します。
実行すると `data/` ディレクトリに複数のレンダリング画像と `cameras.json` が生成されます。

```bash
cargo run --release --bin generate
```

### 2. 学習 (Training)

BurnのAutodiffバックエンド（WGPU）を使用して最適化を実行します。
初期状態の少数のプリミティブから開始し、指定されたStage数に従ってPruningとSplittingを繰り返します。

```bash
cargo run --release --bin train
```

完了後、最適化された物理パラメータが `scene.json` にエクスポートされます。

### 3. リアルタイムビューアー (Viewer)

学習結果の `scene.json` を読み込み、WGPUベースのビューアーで描画します。
学習時と同一のLambertianライティングモデルがWGSL上で評価されます。

```bash
cargo run --release --bin viewer
```

## 🛠 今後の展望 (TODO)

* [ ] COLMAP等の外部SfMツールを用いた、実写画像データセットからの学習パイプライン構築。
* [ ] 球面調和関数 (Spherical Harmonics: SH) の導入による、視点依存の鏡面反射（Specular）表現の対応。
* [ ] 描画パイプラインのラスタライズ化によるフォワードパスの高速化。

## 📝 License

MIT License