const INPUT_JSON = "data/cameras.json";
const OUTPUT_JSON = "data/cameras_filtered.json";

// 何枚ごとに画像をピックアップするか
const INTERVAL = 8;

// targetの距離が平均の何倍離れていたら弾くか（1.5〜2.0くらいがおすすめ）
const OUTLIER_THRESHOLD = 1.5;

function main() {
    const data = JSON.parse(Deno.readTextFileSync(INPUT_JSON));

    // 1. 間引き処理
    const thinned = data.filter((_: any, index: number) => index % INTERVAL === 0);

    // --- 2. 外れ値（異常なカメラ）の除去ロジック ---
    // まず「仮の中心」を計算する
    const initialCenter = thinned.reduce((acc: number[], cam: any) => [
        acc[0] + cam.target[0], acc[1] + cam.target[1], acc[2] + cam.target[2]
    ], [0, 0, 0]).map((v: number) => v / thinned.length);

    // 各カメラのtargetと、仮の中心との「距離」を計算
    const distances = thinned.map((cam: any) => {
        const dx = cam.target[0] - initialCenter[0];
        const dy = cam.target[1] - initialCenter[1];
        const dz = cam.target[2] - initialCenter[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    });

    // 距離の平均を出す
    const avgDistance = distances.reduce((a: number, b: number) => a + b, 0) / distances.length;

    // 平均距離より異常に離れている（OUTLIER_THRESHOLD倍以上）ものを弾く
    const inliers = thinned.filter((_: any, i: number) => distances[i] <= avgDistance * OUTLIER_THRESHOLD);

    const removedCount = thinned.length - inliers.length;
    if (removedCount > 0) {
        console.log(`🚨 警告: 明後日の方向を見ている異常なカメラを ${removedCount} 個弾きました！`);
    }

    // --- 3. 本当の中心を計算（正常なカメラのみを使用） ---
    const finalCenter = inliers.reduce((acc: number[], cam: any) => [
        acc[0] + cam.target[0], acc[1] + cam.target[1], acc[2] + cam.target[2]
    ], [0, 0, 0]).map((v: number) => v / inliers.length);

    // 4. 平行移動 ＆ スケール調整
    const scale = 1.0;

    const processed = inliers.map((cam: any) => {
        return {
            file: cam.file,
            fov: cam.fov,
            origin: [
                (cam.origin[0] - finalCenter[0]) * scale,
                (cam.origin[1] - finalCenter[1]) * scale,
                (cam.origin[2] - finalCenter[2]) * scale,
            ],
            target: [
                (cam.target[0] - finalCenter[0]) * scale,
                (cam.target[1] - finalCenter[1]) * scale,
                (cam.target[2] - finalCenter[2]) * scale,
            ],
        };
    });

    Deno.writeTextFileSync(OUTPUT_JSON, JSON.stringify(processed, null, 2));
    console.log(`✅ Processed ${data.length} -> ${processed.length} valid cameras.`);
    console.log(`🍅 Shifted true center by: [${finalCenter[0].toFixed(2)}, ${finalCenter[1].toFixed(2)}, ${finalCenter[2].toFixed(2)}]`);
    console.log(`📁 Saved to ${OUTPUT_JSON}`);
}

main();