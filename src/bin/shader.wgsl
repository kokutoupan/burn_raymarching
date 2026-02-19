struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    padding: f32,
};

struct Sphere {
    pos: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;

// --- Vertex Shader (Full Screen Quad) ---
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // 2枚の三角形(6頂点)でフルスクリーンの四角形を作る
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), // 左下
        vec2<f32>( 1.0, -1.0), // 右下
        vec2<f32>(-1.0,  1.0), // 左上
        vec2<f32>(-1.0,  1.0), // 左上
        vec2<f32>( 1.0, -1.0), // 右下
        vec2<f32>( 1.0,  1.0)  // 右上
    );

    let p = pos[in_vertex_index];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    
    // Fragment Shader 側のレイマーチング用に -1.0 ~ 1.0 のUVを渡す
    // (Y軸は画面上がプラスになるように反転)
    out.uv = vec2<f32>(p.x, -p.y);
    
    return out;
}

// --- SDF Logic ---
fn smin_exp(a: f32, b: f32, k: f32) -> f32 {
    let res = exp(-k * a) + exp(-k * b);
    return -log(res) / k;
}

fn map(p: vec3<f32>) -> vec2<f32> {
    var d = 10000.0;
    let num_spheres = arrayLength(&spheres);

    for (var i = 0u; i < num_spheres; i = i + 1u) {
        let s = spheres[i];
        let dist = length(p - s.pos) - s.radius;
        if (i == 0u) {
            d = dist;
        } else {
            // Burnコードの soft_min_tensor(all_dists, 32.0) と同じ係数
            d = smin_exp(d, dist, 32.0);
        }
    }
    
    return vec2<f32>(d, 0.0);
}

// 法線計算
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.001;
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * map(p + k.xyy * eps).x +
        k.yyx * map(p + k.yyx * eps).x +
        k.yxy * map(p + k.yxy * eps).x +
        k.xxx * map(p + k.xxx * eps).x
    );
}

// ★追加: renderer.rs と同じ色の加重平均を計算する関数
fn calc_color(p: vec3<f32>) -> vec3<f32> {
    let num_spheres = arrayLength(&spheres);
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var i = 0u; i < num_spheres; i = i + 1u) {
        let s = spheres[i];
        let dist = length(p - s.pos) - s.radius;
        
        // renderer.rsの: weights = dists.mul_scalar(-10.0).exp() を再現
        let w = exp(-dist * 10.0);
        
        color_sum = color_sum + s.color * w;
        weight_sum = weight_sum + w;
    }

    return color_sum / (weight_sum + 0.00001); // 0除算防止
}

// --- Fragment Shader ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    uv.x = uv.x * (uniforms.resolution.x / uniforms.resolution.y);

    // カメラがぐるぐる回る設定
    let radius = 4.0;
    let angle = uniforms.time * 0.5;
    let ro = vec3<f32>(sin(angle) * radius, 0.0, cos(angle) * radius);
    let ta = vec3<f32>(0.0, 0.0, 0.0);

    let ww = normalize(ta - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = normalize(cross(uu, ww));

    let rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);

    // レイマーチング
    var t = 0.0;
    var hit = false;
    for (var i = 0; i < 100; i = i + 1) {
        let p = ro + t * rd;
        let d = map(p).x;
        
        if (d < 0.001) {
            hit = true;
            break;
        }
        if (t > 20.0) {
            break;
        }
        t = t + d;
    }

    // シェーディング
    var col = vec3<f32>(0.0); // 背景黒
    if (hit) {
        let p = ro + t * rd;
        let normal = calc_normal(p);
        
        // ライティング計算
        let light = normalize(vec3<f32>(0.5, 0.8, 1.0));
        let diff = max(dot(normal, light), 0.0);
        let amb = 0.1;
        
        // ★修正: 最も近い球たちから色を合成する
        let obj_color = calc_color(p);
        
        // 物体の色にライティングを掛ける
        col = obj_color * (diff + amb);
    }

    // ガンマ補正 (少し明るく綺麗に見せるため)
    col = pow(col, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(col, 1.0);
}
