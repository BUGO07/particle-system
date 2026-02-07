use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use glam::{EulerRot, Mat4, Quat, Vec3};
use glium::{
    Display, Surface, Texture2d,
    backend::glutin::SimpleWindowBuilder,
    buffer::{Buffer, BufferMode, BufferType},
    glutin::surface::WindowSurface,
    program::ComputeShader,
    uniforms::{ImageUnitAccess, ImageUnitFormat, MagnifySamplerFilter},
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

struct State {
    window: Window,
    display: Display<WindowSurface>,

    render_texture: Texture2d,
    copy_texture: Texture2d,

    particle_shader: ComputeShader,
    ray_shader: ComputeShader,
    copy_shader: ComputeShader,

    particle_buffer: Buffer<[[f32; 4]]>,
    velocity_buffer: Buffer<[[f32; 4]]>,

    cam_pos: Vec3,
    cam_rot: Quat,

    last_update: Instant,
    fixed_time_accum: Duration,

    keys_pressed: HashSet<KeyCode>,
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
        let forward = Vec3::new(local_z.x, 0.0, local_z.z).normalize_or_zero();
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
                    )
                    .unwrap();

                    let mut velocities = self.velocity_buffer.read()?;
                    for _ in 0..count {
                        velocities.push([0.0; 4]);
                    }
                    self.velocity_buffer = Buffer::<[[f32; 4]]>::new(
                        &self.display,
                        &velocities,
                        BufferType::ShaderStorageBuffer,
                        BufferMode::Dynamic,
                    )
                    .unwrap();
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

        let image_unit = self
            .render_texture
            .image_unit(ImageUnitFormat::RGBA8)
            .unwrap()
            .set_access(ImageUnitAccess::Write);

        self.ray_shader.execute(
            uniform! {
                uWidth: self.render_texture.width(),
                uHeight: self.render_texture.height(),
                uCam: Mat4::from_rotation_translation(
                    self.cam_rot,
                    self.cam_pos
                ).to_cols_array_2d(),
                Particles: &self.particle_buffer,
                Velocities: &self.velocity_buffer,
                uDebug: self.keys_pressed.contains(&KeyCode::KeyF),
                uTexture: image_unit,
            },
            self.render_texture.width().div_ceil(16),
            self.render_texture.height().div_ceil(16),
            1,
        );

        let render_unit = self
            .render_texture
            .image_unit(ImageUnitFormat::RGBA8)
            .unwrap()
            .set_access(ImageUnitAccess::Read);

        let final_unit = self
            .copy_texture
            .image_unit(ImageUnitFormat::RGBA8)
            .unwrap()
            .set_access(ImageUnitAccess::Write);

        self.copy_shader.execute(
            uniform! {
                uTexture: render_unit,
                destTexture: final_unit,
            },
            self.render_texture.width(),
            self.render_texture.height(),
            1,
        );

        let frame = self.display.draw();
        self.copy_texture
            .as_surface()
            .fill(&frame, MagnifySamplerFilter::Nearest);
        frame.finish().unwrap();

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

        let render_texture = Texture2d::empty(&display, 1024, 1024).unwrap();
        let copy_texture = Texture2d::empty(&display, 1024, 1024).unwrap();

        let particle_shader =
            ComputeShader::from_source(&display, include_str!("particles.comp")).unwrap();
        let ray_shader =
            ComputeShader::from_source(&display, include_str!("raytracing.comp")).unwrap();
        let copy_shader = ComputeShader::from_source(&display, include_str!("copy.glsl")).unwrap();

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
            render_texture,
            copy_texture,
            particle_shader,
            ray_shader,
            copy_shader,
            particle_buffer,
            velocity_buffer,
            cam_pos: Vec3::new(5.0, 5.0, -15.0),
            cam_rot: Quat::from_euler(EulerRot::YXZ, -15f32.to_radians(), 20f32.to_radians(), 0.0),
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
            yaw += (delta.0 as f32 * scale * 0.00015).to_radians();
            pitch += (delta.1 as f32 * scale * 0.00015).to_radians();
            pitch = pitch.clamp(-1.54, 1.54);

            state.cam_rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
    }
}

fn main() -> anyhow::Result<()> {
    App::new().run()
}
