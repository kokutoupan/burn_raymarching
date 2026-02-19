struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    padding: f32,
    cam_pos: vec3<f32>,
    padding1: f32,
    cam_forward: vec3<f32>,
    padding2: f32,
    cam_up: vec3<f32>,
    padding3: f32,
};

struct Sphere {
    pos: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    let p = pos[in_vertex_index];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, -p.y);
    return out;
}

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
            d = smin_exp(d, dist, 32.0);
        }
    }
    return vec2<f32>(d, 0.0);
}

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

fn calc_color(p: vec3<f32>) -> vec3<f32> {
    let num_spheres = arrayLength(&spheres);
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0u; i < num_spheres; i = i + 1u) {
        let s = spheres[i];
        let dist = length(p - s.pos) - s.radius;
        let w = exp(-dist * 10.0);
        color_sum = color_sum + s.color * w;
        weight_sum = weight_sum + w;
    }
    return color_sum / (weight_sum + 0.00001);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    uv.x = uv.x * (uniforms.resolution.x / uniforms.resolution.y);

    // Rust側から受け取ったカメラ姿勢を適用
    let ro = uniforms.cam_pos;
    let ww = normalize(uniforms.cam_forward);
    let uu = normalize(cross(ww, uniforms.cam_up));
    let vv = normalize(cross(uu, ww));
    let rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < 100; i = i + 1) {
        let p = ro + t * rd;
        let d = map(p).x;
        if (d < 0.001) { hit = true; break; }
        if (t > 20.0) { break; }
        t = t + d;
    }

    var col = vec3<f32>(0.0);
    if (hit) {
        let p = ro + t * rd;
        let normal = calc_normal(p);
        
        // renderer.rsのライト設定に完全一致させる [-0.5, 0.5, -1.0]
        let light = normalize(vec3<f32>(-0.5, 0.5, -1.0));
        let diff = max(dot(normal, light), 0.0);
        
        let obj_color = calc_color(p);
        
        // lighting = diffuse + 0.1
        col = obj_color * (diff + 0.1);
    }

    return vec4<f32>(col, 1.0);
}
