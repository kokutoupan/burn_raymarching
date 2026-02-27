const OUTPUT_JSON = "data/cameras_filtered.json"; // Rustに読ませる用
const IMAGE_DIR = "data/tomato/images_nobg";

// 画像フォルダからファイル一覧を取得してソート
const files = Array.from(Deno.readDirSync(IMAGE_DIR))
    .filter((f) => f.name.endsWith(".jpg"))
    .map((f) => f.name)
    .sort();

// ---- パラメータ設定 ----
// 動画から抽出した画像のうち、学習に使う枚数（間引き率）
// 120枚あるなら、INTERVAL=6 にすると 20枚 になります。
const INTERVAL = 24;
const filteredFiles = files.filter((_, i) => i % INTERVAL === 0);
const numCameras = filteredFiles.length;

// カメラの軌跡（円）の半径と高さ
const RADIUS = 2.0; // トマトからの距離（2.0くらいがちょうどいいです）
const HEIGHT = 0.5; // カメラの高さ（少し上から見下ろす）
const FOV = 50.0;   // 視野角

const processed = filteredFiles.map((filename, i) => {
    // 1周 (2 * PI) を均等に分割
    const angle = (i / numCameras) * Math.PI * 2.0;

    // origin (カメラ位置): 円周上の座標
    const x = Math.cos(angle) * RADIUS;
    const z = Math.sin(angle) * RADIUS;
    const origin = [x, HEIGHT, z];

    // トマトは原点 [0, 0, 0] にいる。
    // target は、originからトマト(原点)へ向かうベクトルの先（長さ1）
    const dirX = -x;
    const dirY = -HEIGHT;
    const dirZ = -z;
    const len = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);

    const target = [
        origin[0] + dirX / len,
        origin[1] + dirY / len,
        origin[2] + dirZ / len,
    ];

    return {
        file: `data/tomato/images_nobg/${filename}`,
        fov: FOV,
        origin: origin,
        target: target,
    };
});

Deno.writeTextFileSync(OUTPUT_JSON, JSON.stringify(processed, null, 2));
console.log(`✅ ${numCameras} 枚の完璧な円軌道カメラを捏造しました！`);
console.log(`📁 Saved to ${OUTPUT_JSON}`);