// colmap_to_json.ts
const COLMAP_DIR = "data/tomato/sparse/0";
const OUTPUT_JSON = "data/cameras.json";
const IMAGE_PREFIX = "data/tomato/images_nobg/";

// クォータニオンから回転行列(3x3)への変換
function qvec2rotmat(qw: number, qx: number, qy: number, qz: number): number[][] {
    return [
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ];
}

// 3x3行列の転置
function transpose(m: number[][]): number[][] {
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]]
    ];
}

// 行列(3x3)とベクトル(3)の積
function dot(m: number[][], v: number[]): number[] {
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ];
}

function main() {
    const camerasText = Deno.readTextFileSync(`${COLMAP_DIR}/cameras.txt`);
    const imagesText = Deno.readTextFileSync(`${COLMAP_DIR}/images.txt`);

    // カメラFOVのパース
    const fovDict: Record<number, number> = {};
    for (const line of camerasText.split("\n")) {
        if (line.startsWith("#") || line.trim() === "") continue;
        const elems = line.split(" ");
        const camId = parseInt(elems[0]);
        const height = parseFloat(elems[3]);
        const focalLength = parseFloat(elems[4]);

        // FOV = 2 * atan(height / (2 * focal))
        const fovY = 2.0 * Math.atan(height / (2.0 * focalLength));
        fovDict[camId] = fovY * (180.0 / Math.PI); // 度数法に変換
    }

    // 画像ポーズのパース
    const cameraConfigs = [];
    const imageLines = imagesText.split("\n");

    for (let i = 0; i < imageLines.length; i += 2) {
        const line = imageLines[i].trim();
        if (line.startsWith("#") || line === "") continue;

        const elems = line.split(" ");
        const qw = parseFloat(elems[1]);
        const qx = parseFloat(elems[2]);
        const qy = parseFloat(elems[3]);
        const qz = parseFloat(elems[4]);
        const tx = parseFloat(elems[5]);
        const ty = parseFloat(elems[6]);
        const tz = parseFloat(elems[7]);
        const camId = parseInt(elems[8]);
        const imageName = elems[9];

        // COLMAPの姿勢を計算
        const R = qvec2rotmat(qw, qx, qy, qz);
        const Rt = transpose(R);
        const tvec = [tx, ty, tz];

        // カメラ中心座標: origin = -R^T * t
        const origin = dot(Rt, tvec).map(v => -v);

        // 視線方向ベクトル: R^T * [0, 0, 1]
        const lookDir = dot(Rt, [0.0, 0.0, 1.0]);

        // Target = origin + lookDir
        const target = [
            origin[0] + lookDir[0],
            origin[1] + lookDir[1],
            origin[2] + lookDir[2]
        ];

        cameraConfigs.push({
            file: `${IMAGE_PREFIX}${imageName}`,
            origin: origin,
            target: target,
            fov: fovDict[camId]
        });
    }

    Deno.writeTextFileSync(OUTPUT_JSON, JSON.stringify(cameraConfigs, null, 2));
    console.log(`✅ Generated ${cameraConfigs.length} camera configs to ${OUTPUT_JSON}`);
}

main();