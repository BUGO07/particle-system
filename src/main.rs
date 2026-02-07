use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use glam::{EulerRot, Mat4, Quat, Vec3};
use glium::{
    Display, DrawParameters, Surface,
    backend::glutin::SimpleWindowBuilder,
    buffer::{Buffer, BufferMode, BufferType},
    glutin::surface::WindowSurface,
    program::ComputeShader,
    winit::{
        application::ApplicationHandler,
        dpi::PhysicalSize,
        event::{DeviceEvent, DeviceId, WindowEvent},
        event_loop::{ActiveEventLoop, EventLoop},
        keyboard::{KeyCode, PhysicalKey},
        window::{CursorGrabMode, Window, WindowAttributes, WindowId},
    },
};

#[macro_use]
extern crate glium;

const PARTICLE_COUNT: usize = 1000;
const PARTICLE_SIZE: f32 = 0.2;

#[derive(Copy, Clone)]
struct SphereVertex {
    position: [f32; 3],
}
implement_vertex!(SphereVertex, position);

#[derive(Copy, Clone)]
struct ParticleData {
    particle: [f32; 4], // xyz = center, w = radius
}
implement_vertex!(ParticleData, particle);

struct State {
    window: Window,
    display: Display<WindowSurface>,

    particle_shader: ComputeShader,

    particle_buffer: Buffer<[[f32; 4]]>,
    velocity_buffer: Buffer<[[f32; 4]]>,

    cam_pos: Vec3,
    cam_rot: Quat,

    last_update: Instant,
    fixed_time_accum: Duration,

    keys_pressed: HashSet<KeyCode>,
}

fn create_sphere_mesh(
    display: &glium::Display<WindowSurface>,
    subdivisions: u32,
) -> (glium::VertexBuffer<SphereVertex>, glium::IndexBuffer<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Create a simple UV sphere
    for lat in 0..=subdivisions {
        let theta = lat as f32 * std::f32::consts::PI / subdivisions as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=subdivisions {
            let phi = lon as f32 * 2.0 * std::f32::consts::PI / subdivisions as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            vertices.push(SphereVertex {
                position: [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
            });
        }
    }

    // Generate indices
    for lat in 0..subdivisions {
        for lon in 0..subdivisions {
            let first = lat * (subdivisions + 1) + lon;
            let second = first + subdivisions + 1;

            indices.push(first);
            indices.push(second);
            indices.push(first + 1);

            indices.push(second);
            indices.push(second + 1);
            indices.push(first + 1);
        }
    }

    let vertex_buffer = glium::VertexBuffer::new(display, &vertices).unwrap();
    let index_buffer = glium::IndexBuffer::new(
        display,
        glium::index::PrimitiveType::TrianglesList,
        &indices,
    )
    .unwrap();

    (vertex_buffer, index_buffer)
}

impl State {
    fn resize(&mut self, width: u32, height: u32) {
        self.display.resize((width, height));
    }

    fn update(&mut self) -> anyhow::Result<()> {
        println!("Rendering {} Particles", self.particle_buffer.len());
        let delta = self.last_update.elapsed().as_secs_f32();
        self.last_update = Instant::now();

        let local_z = self.cam_rot * Vec3::Z;
        let forward = -Vec3::new(local_z.x, 0.0, local_z.z).normalize_or_zero();
        let right = Vec3::new(local_z.z, 0.0, -local_z.x).normalize_or_zero();
        let up = Vec3::Y;

        let mut move_dir = Vec3::ZERO;
        let mut speed = 5.0;

        for key in &self.keys_pressed {
            match key {
                KeyCode::ControlLeft => speed *= 10.0,
                KeyCode::KeyW => move_dir += forward,
                KeyCode::KeyS => move_dir -= forward,
                KeyCode::KeyA => move_dir -= right,
                KeyCode::KeyD => move_dir += right,
                KeyCode::Space => move_dir += up,
                KeyCode::ShiftLeft => move_dir -= up,
                KeyCode::KeyH => {
                    let count = if self.keys_pressed.contains(&KeyCode::ControlLeft) {
                        10
                    } else {
                        1
                    };
                    let mut particles = self.particle_buffer.read()?;
                    for _ in 0..count {
                        particles.push([
                            rand::random_range(-5.0..5.0),
                            rand::random_range(4.0..5.0),
                            rand::random_range(-5.0..5.0),
                            PARTICLE_SIZE,
                        ]);
                    }
                    self.particle_buffer = Buffer::<[[f32; 4]]>::new(
                        &self.display,
                        &particles,
                        BufferType::ShaderStorageBuffer,
                        BufferMode::Dynamic,
                    )?;

                    let mut velocities = self.velocity_buffer.read()?;
                    for _ in 0..count {
                        velocities.push([0.0; 4]);
                    }
                    self.velocity_buffer = Buffer::<[[f32; 4]]>::new(
                        &self.display,
                        &velocities,
                        BufferType::ShaderStorageBuffer,
                        BufferMode::Dynamic,
                    )?;
                }
                _ => {}
            }
        }

        self.cam_pos += move_dir.normalize_or_zero() * speed * delta;

        self.fixed_time_accum += Duration::from_secs_f32(delta);
        let fixed_delta = Duration::from_secs_f32(1.0 / 60.0);
        while self.fixed_time_accum >= fixed_delta {
            self.fixed_time_accum -= fixed_delta;

            self.particle_shader.execute(
                uniform! {
                    uDelta: delta,
                    uForce: self.keys_pressed.contains(&KeyCode::KeyG),
                    uCam: Mat4::from_rotation_translation(
                        self.cam_rot,
                        self.cam_pos
                    ).to_cols_array_2d(),
                    Particles: &self.particle_buffer,
                    Velocities: &self.velocity_buffer,
                },
                self.particle_buffer.len().div_ceil(1024) as u32,
                1,
                1,
            );
        }

        let mut frame = self.display.draw();
        frame.clear_color_and_depth((0.0, 0.0, 1.0, 1.0), 1.0);
        let (vertex_buffer, index_buffer) = create_sphere_mesh(&self.display, 8);

        let instance_buffer: glium::VertexBuffer<ParticleData> =
            glium::VertexBuffer::empty(&self.display, self.particle_buffer.len()).unwrap();
        instance_buffer.write(
            self.particle_buffer
                .read()?
                .into_iter()
                .map(|particle| ParticleData { particle })
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let program = glium::Program::from_source(
            &self.display,
            include_str!("vert.glsl"),
            include_str!("frag.glsl"),
            None,
        )
        .unwrap();

        let (width, height) = frame.get_dimensions();

        frame
            .draw(
                (&vertex_buffer, instance_buffer.per_instance().unwrap()),
                &index_buffer,
                &program,
                &uniform! {
                    projection: Mat4::perspective_rh_gl(
                        45f32.to_radians(),
                        width as f32 / height as f32,
                        0.1,
                        100.0
                    ).to_cols_array_2d(),
                    view: Mat4::from_rotation_translation(
                        self.cam_rot,
                        self.cam_pos
                    ).inverse().to_cols_array_2d(),
                    uCamPos: self.cam_pos.to_array(),
                },
                &DrawParameters {
                    depth: glium::Depth {
                        test: glium::draw_parameters::DepthTest::IfLess,
                        write: true,
                        ..Default::default()
                    },
                    point_size: Some(10.0),
                    ..Default::default()
                },
            )
            .unwrap();
        frame.finish()?;

        self.window.request_redraw();

        Ok(())
    }
}

struct App {
    state: Option<State>,
    window_attributes: WindowAttributes,
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            window_attributes: WindowAttributes::default()
                .with_title("GPU Particle Ray Tracer")
                .with_inner_size(PhysicalSize::new(1280, 720)),
        }
    }

    fn run(mut self) -> anyhow::Result<()> {
        EventLoop::with_user_event().build()?.run_app(&mut self)?;
        Ok(())
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, display) = SimpleWindowBuilder::new()
            .set_window_builder(self.window_attributes.clone())
            .build(event_loop);

        let particle_shader =
            ComputeShader::from_source(&display, include_str!("compute.glsl")).unwrap();

        window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        window.set_cursor_visible(false);

        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        for _ in 0..PARTICLE_COUNT {
            particles.push([
                rand::random_range(-5.0..5.0),
                rand::random_range(-5.0..5.0),
                rand::random_range(-5.0..5.0),
                PARTICLE_SIZE,
            ]);
        }

        let particle_buffer = Buffer::<[[f32; 4]]>::new(
            &display,
            &particles,
            BufferType::ShaderStorageBuffer,
            BufferMode::Dynamic,
        )
        .unwrap();

        let velocity_buffer = Buffer::<[[f32; 4]]>::new(
            &display,
            &[[0.0; 4]; PARTICLE_COUNT],
            BufferType::ShaderStorageBuffer,
            BufferMode::Dynamic,
        )
        .unwrap();

        self.state = Some(State {
            window,
            display,
            particle_shader,
            particle_buffer,
            velocity_buffer,
            cam_pos: Vec3::new(5.0, 5.0, 15.0),
            cam_rot: Quat::from_euler(EulerRot::YXZ, 15f32.to_radians(), -20f32.to_radians(), 0.0),
            last_update: Instant::now(),
            fixed_time_accum: Duration::ZERO,
            keys_pressed: HashSet::new(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match self.state {
            Some(ref mut s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.update().unwrap(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state.is_pressed() {
                        if code == KeyCode::Escape {
                            state.window.set_cursor_visible(true);
                            state.window.set_cursor_grab(CursorGrabMode::None).unwrap();
                        }
                        state.keys_pressed.insert(code);
                    } else {
                        state.keys_pressed.remove(&code);
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let state = match self.state {
                Some(ref mut s) => s,
                None => return,
            };

            let size = state.window.inner_size();
            let scale = size.height.max(size.width) as f32;

            let (mut yaw, mut pitch, _) = state.cam_rot.to_euler(EulerRot::YXZ);
            yaw -= (delta.0 as f32 * scale * 0.00015).to_radians();
            pitch -= (delta.1 as f32 * scale * 0.00015).to_radians();
            pitch = pitch.clamp(-1.54, 1.54);

            state.cam_rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
    }
}

fn main() -> anyhow::Result<()> {
    App::new().run()
}
