use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId}; // 数学ライブラリを使用

// --- データ構造 ---
#[derive(serde::Deserialize)]
struct SceneData {
    num_spheres: usize,
    centers: Vec<f32>,
    colors: Vec<f32>,
    radii: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereRaw {
    pos: [f32; 3],
    radius: f32,
    color: [f32; 3],
    padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    padding: f32,
    cam_pos: [f32; 3],
    padding1: f32,
    cam_forward: [f32; 3],
    padding2: f32,
    cam_up: [f32; 3],
    padding3: f32,
}

// --- 入力とカメラ ---
#[derive(Default)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    turn_left: bool,
    turn_right: bool,
    turn_up: bool,
    turn_down: bool,
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }
    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }
    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }
}

// --- アプリケーションの状態 ---
struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    sphere_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start_time: std::time::Instant,
    last_update: std::time::Instant,
    uniforms: Uniforms,
    camera: Camera,
    input: InputState,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Pure Wgpu Raymarching (WASD Camera)")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 800.0));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());

            let state = pollster::block_on(State::new(window.clone()));
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match self.state.as_mut() {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: key_state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = key_state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => state.input.forward = is_pressed,
                    KeyCode::KeyS => state.input.backward = is_pressed,
                    KeyCode::KeyA => state.input.left = is_pressed,
                    KeyCode::KeyD => state.input.right = is_pressed,
                    KeyCode::KeyE | KeyCode::Space => state.input.up = is_pressed,
                    KeyCode::KeyQ | KeyCode::ShiftLeft => state.input.down = is_pressed,
                    KeyCode::ArrowLeft => state.input.turn_left = is_pressed,
                    KeyCode::ArrowRight => state.input.turn_right = is_pressed,
                    KeyCode::ArrowUp => state.input.turn_up = is_pressed,
                    KeyCode::ArrowDown => state.input.turn_down = is_pressed,
                    KeyCode::Escape => {
                        if is_pressed {
                            event_loop.exit()
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::RedrawRequested => {
                state.update();
                if let Some(window) = self.window.as_ref() {
                    let pos = state.camera.pos;
                    // ラジアンから度数法(Degrees)に変換すると直感的にわかりやすいです
                    let yaw = state.camera.yaw.to_degrees();
                    let pitch = state.camera.pitch.to_degrees();

                    window.set_title(&format!(
                        "Viewer | Pos: ({:.2}, {:.2}, {:.2}) | Yaw: {:.0}°, Pitch: {:.0}°",
                        pos.x, pos.y, pos.z, yaw, pitch
                    ));
                }
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }

                // ★追加: 60FPS固定（GPU負荷の抑制）
                let target_frametime = std::time::Duration::from_secs_f64(1.0 / 60.0);
                let elapsed = state.last_update.elapsed();
                if elapsed < target_frametime {
                    // 16.6ms 経っていなければ、残りの時間はスレッドを休ませる
                    std::thread::sleep(target_frametime - elapsed);
                }

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let file = File::open("scene.json").expect("scene.json not found. Run train first.");
        let reader = BufReader::new(file);
        let scene: SceneData = serde_json::from_reader(reader).expect("JSON parse error");

        let mut spheres = Vec::new();
        for i in 0..scene.num_spheres {
            spheres.push(SphereRaw {
                pos: [
                    scene.centers[i * 3],
                    scene.centers[i * 3 + 1],
                    scene.centers[i * 3 + 2],
                ],
                radius: scene.radii[i],
                color: [
                    scene.colors[i * 3],
                    scene.colors[i * 3 + 1],
                    scene.colors[i * 3 + 2],
                ],
                padding: 0.0,
            });
        }

        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Buffer"),
            contents: bytemuck::cast_slice(&spheres),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // カメラの初期位置 (z=-2.5から +Z方向を見る)
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, -2.5),
            yaw: std::f32::consts::PI / 2.0,
            pitch: 0.0,
        };

        let uniforms = Uniforms {
            resolution: [size.width as f32, size.height as f32],
            time: 0.0,
            padding: 0.0,
            cam_pos: camera.pos.into(),
            padding1: 0.0,
            cam_forward: camera.forward().into(),
            padding2: 0.0,
            cam_up: camera.up().into(),
            padding3: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sphere_buffer.as_entire_binding(),
                },
            ],
            label: Some("bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            sphere_buffer,
            uniform_buffer,
            bind_group,
            start_time: std::time::Instant::now(),
            last_update: std::time::Instant::now(),
            uniforms,
            camera,
            input: InputState::default(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.uniforms.resolution = [new_size.width as f32, new_size.height as f32];
        }
    }

    fn update(&mut self) {
        let now = std::time::Instant::now();
        let dt = (now - self.last_update).as_secs_f32();
        self.last_update = now;

        // カメラ移動と回転の速度
        let move_speed = 3.0 * dt;
        let turn_speed = 1.5 * dt;

        let forward = self.camera.forward();
        let right = self.camera.right();
        let up = Vec3::Y; // 上下移動は常にワールドY軸とする

        if self.input.forward {
            self.camera.pos += forward * move_speed;
        }
        if self.input.backward {
            self.camera.pos -= forward * move_speed;
        }
        if self.input.left {
            self.camera.pos -= right * move_speed;
        }
        if self.input.right {
            self.camera.pos += right * move_speed;
        }
        if self.input.up {
            self.camera.pos += up * move_speed;
        }
        if self.input.down {
            self.camera.pos -= up * move_speed;
        }

        if self.input.turn_left {
            self.camera.yaw -= turn_speed;
        }
        if self.input.turn_right {
            self.camera.yaw += turn_speed;
        }
        if self.input.turn_up {
            self.camera.pitch += turn_speed;
        }
        if self.input.turn_down {
            self.camera.pitch -= turn_speed;
        }

        // 真上・真下を見すぎないようにクランプ (-89度 ～ 89度)
        self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55);

        self.uniforms.time = self.start_time.elapsed().as_secs_f32();
        self.uniforms.cam_pos = self.camera.pos.into();
        self.uniforms.cam_forward = self.camera.forward().into();
        self.uniforms.cam_up = self.camera.up().into();

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
