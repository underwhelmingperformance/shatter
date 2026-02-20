use shatter_macros::progress;
use tracing::{info, warn};
use wgpu::util::DeviceExt;

use crate::PanelDimensions;
use crate::media::ImagePreprocessor;

use super::service::fps_to_delay_centiseconds;
use super::{RenderReceipt, TransitionError, TransitionRequest};

const MAX_CHUNK_FRAMES: usize = 64;
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

// Algorithm note:
// This shader simulates an explosion behind the image plane. All cell-sized
// debris flies forward towards the camera, scaling up via perspective projection.
// Cells further from centre drift off-screen faster (parallax), while centre
// cells loom large before fading. Edges crack first, the centre holds longest.
// Once the source image has fully dispersed, the process reverses to assemble
// the target image.
const SHATTER_SHADER: &str = r#"
struct Params {
    out_width: u32,
    out_height: u32,
    from_width: u32,
    from_height: u32,
    to_width: u32,
    to_height: u32,
    total_frames: u32,
    frame_start: u32,
    chunk_frames: u32,
    hold_frames: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> from_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read> to_pixels: array<u32>;

@group(0) @binding(2)
var<storage, read_write> out_pixels: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

const GRID_X: f32 = 20.0;
const GRID_Y: f32 = 20.0;
const SMOOTHNESS: f32 = 0.08;

fn unpack_rgba(pixel: u32) -> vec4<f32> {
    let r = f32(pixel & 0xFFu) / 255.0;
    let g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    let b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    let a = f32((pixel >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(pixel: vec4<f32>) -> u32 {
    let clamped = clamp(pixel, vec4<f32>(0.0), vec4<f32>(1.0));
    let r = u32(round(clamped.r * 255.0));
    let g = u32(round(clamped.g * 255.0));
    let b = u32(round(clamped.b * 255.0));
    let a = u32(round(clamped.a * 255.0));
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

fn hash01_u32(x: u32, y: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    var state = x * 374761393u + y * 668265263u + seed_lo * 982451653u + seed_hi * 2654435761u;
    state = (state ^ (state >> 13u)) * 1274126177u;
    state = state ^ (state >> 16u);
    return f32(state & 0x00FFFFFFu) / 16777215.0;
}

fn hash_channel(cx: u32, cy: u32, ch: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    return hash01_u32(cx + ch * 7919u, cy + ch * 6271u, seed_lo, seed_hi);
}

fn cell_progress(cell_noise: f32, phase_progress: f32) -> f32 {
    let edge0 = max(0.0, cell_noise - SMOOTHNESS);
    let edge1 = min(1.0, cell_noise + SMOOTHNESS);
    return smoothstep(edge0, edge1, phase_progress);
}

fn ease_out_quart(x: f32) -> f32 {
    let t = 1.0 - x;
    return 1.0 - t * t * t * t;
}

fn ease_in_cubic(x: f32) -> f32 {
    return x * x * x;
}

fn timeline_progress(frame_index: u32) -> f32 {
    if (params.total_frames <= 1u) {
        return 1.0;
    }

    let max_hold = (params.total_frames - 1u) / 2u;
    let hold = min(params.hold_frames, max_hold);
    if (frame_index < hold) {
        return 0.0;
    }

    let final_transition_frame = params.total_frames - hold - 1u;
    if (frame_index >= final_transition_frame) {
        return 1.0;
    }

    let transition_frames = params.total_frames - (hold * 2u);
    if (transition_frames <= 1u) {
        return 1.0;
    }

    let active_index = frame_index - hold;
    return f32(active_index) / f32(transition_frames - 1u);
}

fn fit_sample_from(uv: vec2<f32>) -> vec4<f32> {
    let out_w = f32(params.out_width);
    let out_h = f32(params.out_height);
    let src_w = f32(params.from_width);
    let src_h = f32(params.from_height);

    let scale = min(out_w / src_w, out_h / src_h);
    let scaled_w = src_w * scale;
    let scaled_h = src_h * scale;
    let offset_x = (out_w - scaled_w) * 0.5;
    let offset_y = (out_h - scaled_h) * 0.5;

    let px = uv.x * out_w;
    let py = uv.y * out_h;

    if (px < offset_x || py < offset_y || px >= offset_x + scaled_w || py >= offset_y + scaled_h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let u = (px - offset_x) / scaled_w;
    let v = (py - offset_y) / scaled_h;

    let sx = min(u32(floor(u * src_w)), params.from_width - 1u);
    let sy = min(u32(floor(v * src_h)), params.from_height - 1u);
    let index = sy * params.from_width + sx;
    return unpack_rgba(from_pixels[index]);
}

fn fit_sample_to(uv: vec2<f32>) -> vec4<f32> {
    let out_w = f32(params.out_width);
    let out_h = f32(params.out_height);
    let src_w = f32(params.to_width);
    let src_h = f32(params.to_height);

    let scale = min(out_w / src_w, out_h / src_h);
    let scaled_w = src_w * scale;
    let scaled_h = src_h * scale;
    let offset_x = (out_w - scaled_w) * 0.5;
    let offset_y = (out_h - scaled_h) * 0.5;

    let px = uv.x * out_w;
    let py = uv.y * out_h;

    if (px < offset_x || py < offset_y || px >= offset_x + scaled_w || py >= offset_y + scaled_h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let u = (px - offset_x) / scaled_w;
    let v = (py - offset_y) / scaled_h;

    let sx = min(u32(floor(u * src_w)), params.to_width - 1u);
    let sy = min(u32(floor(v * src_h)), params.to_height - 1u);
    let index = sy * params.to_width + sx;
    return unpack_rgba(to_pixels[index]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_width || gid.y >= params.out_height || gid.z >= params.chunk_frames) {
        return;
    }

    let pixel_index = gid.y * params.out_width + gid.x;
    let frame_index = params.frame_start + gid.z;

    let progress = timeline_progress(frame_index);

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(params.out_width),
        (f32(gid.y) + 0.5) / f32(params.out_height)
    );

    let cell = floor(vec2<f32>(GRID_X, GRID_Y) * uv);
    let cx = u32(cell.x);
    let cy = u32(cell.y);
    let cell_center = (cell + 0.5) / vec2<f32>(GRID_X, GRID_Y);

    let r_base  = hash01_u32(cx, cy, params.seed_lo, params.seed_hi);
    let r_angle = hash_channel(cx, cy, 1u, params.seed_lo, params.seed_hi);
    let r_speed = hash_channel(cx, cy, 3u, params.seed_lo, params.seed_hi);

    // Forward speed: how fast this cell flies towards the camera.
    let speed = 0.6 + r_speed * 0.8;

    // Subtle lateral drift direction (random per cell).
    let drift_angle = r_angle * 6.283185;
    let drift_dir = vec2<f32>(cos(drift_angle), sin(drift_angle));

    // Wave ordering: edges crack first, centre holds longest.
    let centre_dist = length(cell_center - vec2<f32>(0.5, 0.5));
    let centre_factor = clamp(centre_dist / 0.707, 0.0, 1.0);
    let order = r_base * 0.5 + (1.0 - centre_factor) * 0.5;

    // Perspective vanishing point.
    let vp = vec2<f32>(0.5, 0.5);

    let out_index = gid.z * (params.out_width * params.out_height) + pixel_index;

    // Stage one: explosion behind the image pushes cells towards the camera.
    if (progress < 0.5) {
        let phase = progress * 2.0;
        let eased = ease_out_quart(phase);

        let disappear = cell_progress(order, phase);

        // Perspective scale: cell approaches camera along Z.
        let z = eased * speed;
        let scale = 1.0 / max(1.0 - z, 0.01);

        // Subtle lateral scatter.
        let drift = drift_dir * eased * speed * 0.06;

        let sample_uv = clamp(
            vp + (uv - vp - drift) / scale,
            vec2<f32>(0.0), vec2<f32>(1.0)
        );
        let from_color = fit_sample_from(sample_uv);
        let alpha = from_color.a * (1.0 - disappear) / scale;

        if (alpha <= 0.0) {
            out_pixels[out_index] = pack_rgba(vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
        out_pixels[out_index] = pack_rgba(vec4<f32>(from_color.rgb, alpha));
        return;
    }

    // Stage two: target image assembles from scattered state.
    let phase = (progress - 0.5) * 2.0;
    let eased = ease_in_cubic(phase);
    let remaining = 1.0 - eased;

    let appear = cell_progress(order, eased);

    let z = remaining * speed;
    let scale = 1.0 / max(1.0 - z, 0.01);

    let drift = drift_dir * remaining * speed * 0.06;

    let sample_uv = clamp(
        vp + (uv - vp - drift) / scale,
        vec2<f32>(0.0), vec2<f32>(1.0)
    );
    let to_color = fit_sample_to(sample_uv);
    let alpha = to_color.a * appear / scale;

    if (alpha <= 0.0) {
        out_pixels[out_index] = pack_rgba(vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    out_pixels[out_index] = pack_rgba(vec4<f32>(to_color.rgb, alpha));
}
"#;

#[derive(Debug, Default)]
pub(super) struct ShatterTransitionRenderer;

impl ShatterTransitionRenderer {
    pub(super) fn new() -> Self {
        Self
    }

    #[progress(
        message = "Rendering shatter transition",
        finished = match result {
            Ok(_receipt) => "Shatter render complete".to_string(),
            Err(_error) => "Shatter render failed".to_string(),
        },
        skip(self, request),
        level = "info"
    )]
    pub(super) fn render(
        &self,
        request: &TransitionRequest,
    ) -> Result<RenderReceipt, TransitionError> {
        let from_image = ImagePreprocessor::decode_oriented_from_path(request.from_path())
            .map_err(|source| TransitionError::SourceImage {
                path: request.from_path().to_path_buf(),
                source,
            })?;
        let to_image =
            ImagePreprocessor::decode_oriented_from_path(request.to_path()).map_err(|source| {
                TransitionError::SourceImage {
                    path: request.to_path().to_path_buf(),
                    source,
                }
            })?;

        let from_width =
            u16::try_from(from_image.width()).map_err(|_error| TransitionError::GpuFailure {
                reason: "source image width exceeds u16 range".to_string(),
            })?;
        let from_height =
            u16::try_from(from_image.height()).map_err(|_error| TransitionError::GpuFailure {
                reason: "source image height exceeds u16 range".to_string(),
            })?;
        let to_width =
            u16::try_from(to_image.width()).map_err(|_error| TransitionError::GpuFailure {
                reason: "target image width exceeds u16 range".to_string(),
            })?;
        let to_height =
            u16::try_from(to_image.height()).map_err(|_error| TransitionError::GpuFailure {
                reason: "target image height exceeds u16 range".to_string(),
            })?;
        let from_dimensions = PanelDimensions::new(from_width, from_height).ok_or_else(|| {
            TransitionError::GpuFailure {
                reason: "source image dimensions must be non-zero".to_string(),
            }
        })?;
        let to_dimensions = PanelDimensions::new(to_width, to_height).ok_or_else(|| {
            TransitionError::GpuFailure {
                reason: "target image dimensions must be non-zero".to_string(),
            }
        })?;
        let dimensions = request.size().resolve(from_dimensions, to_dimensions);
        let out_width = dimensions.width();
        let out_height = dimensions.height();

        let from_pixels = pack_rgba_to_u32(from_image.as_raw());
        let to_pixels = pack_rgba_to_u32(to_image.as_raw());

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let (adapter, used_fallback_adapter) = request_adapter(&instance)?;
        if used_fallback_adapter {
            warn!("hardware adapter unavailable; using fallback adapter");
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("shatter-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| TransitionError::GpuFailure {
            reason: format!("request_device failed: {error}"),
        })?;

        let from_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("from-buffer"),
            contents: bytemuck::cast_slice(&from_pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let to_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("to-buffer"),
            contents: bytemuck::cast_slice(&to_pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let total_frames = request.frame_count().get();
        let pixel_count = usize::from(out_width) * usize::from(out_height);
        let bytes_per_frame = pixel_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| TransitionError::GpuFailure {
                reason: "frame byte size overflow".to_string(),
            })?;

        let chunk_frames = compute_chunk_frames(&device, total_frames, bytes_per_frame)?;
        let output_buffer_size =
            u64::try_from(chunk_frames * bytes_per_frame).map_err(|_error| {
                TransitionError::GpuFailure {
                    reason: "output buffer size overflow".to_string(),
                }
            })?;
        let staging_buffer_size = align_to(output_buffer_size, wgpu::MAP_ALIGNMENT);

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer"),
            size: staging_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params-buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shatter-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shatter-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shatter-shader"),
            source: wgpu::ShaderSource::Wgsl(SHATTER_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shatter-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shatter-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: from_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: to_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut output_file = std::fs::File::create(request.output_path()).map_err(|source| {
            TransitionError::OutputIo {
                path: request.output_path().to_path_buf(),
                source,
            }
        })?;
        let mut encoder = gif::Encoder::new(&mut output_file, out_width, out_height, &[])
            .map_err(|source| TransitionError::GifEncoding { source })?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(|source| TransitionError::GifEncoding { source })?;

        let delay = fps_to_delay_centiseconds(request.fps().get());
        let frames_total = usize::from(total_frames);

        progress_set_length!(frames_total);
        let mut frame_start = 0usize;
        while frame_start < frames_total {
            let chunk_len = (frames_total - frame_start).min(chunk_frames);

            let params = [
                u32::from(out_width),
                u32::from(out_height),
                u32::from(from_width),
                u32::from(from_height),
                u32::from(to_width),
                u32::from(to_height),
                u32::from(total_frames),
                u32::try_from(frame_start).map_err(|_error| TransitionError::GpuFailure {
                    reason: "frame start overflow".to_string(),
                })?,
                u32::try_from(chunk_len).map_err(|_error| TransitionError::GpuFailure {
                    reason: "chunk frame count overflow".to_string(),
                })?,
                u32::from(request.hold_frames()),
                request.seed() as u32,
                (request.seed() >> 32) as u32,
                0,
                0,
                0,
            ];
            queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shatter-command-encoder"),
                });
            {
                let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shatter-compute-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    div_ceil_u32(u32::from(out_width), WORKGROUP_SIZE_X),
                    div_ceil_u32(u32::from(out_height), WORKGROUP_SIZE_Y),
                    u32::try_from(chunk_len).map_err(|_error| TransitionError::GpuFailure {
                        reason: "dispatch chunk overflow".to_string(),
                    })?,
                );
            }

            let chunk_bytes = u64::try_from(chunk_len * bytes_per_frame).map_err(|_error| {
                TransitionError::GpuFailure {
                    reason: "chunk byte size overflow".to_string(),
                }
            })?;
            command_encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                0,
                chunk_bytes,
            );
            queue.submit(Some(command_encoder.finish()));

            let mapped_bytes = map_staging_chunk(&device, &staging_buffer, chunk_bytes)?;
            let words = bytemuck::cast_slice::<u8, u32>(&mapped_bytes);
            for local_index in 0..chunk_len {
                let start = local_index * pixel_count;
                let end = start + pixel_count;
                let frame_words = &words[start..end];
                let mut rgba = Vec::with_capacity(pixel_count * 4);
                for &packed in frame_words {
                    rgba.extend_from_slice(&packed.to_le_bytes());
                }

                let mut frame = gif::Frame::from_rgba_speed(out_width, out_height, &mut rgba, 10);
                frame.delay = delay;
                frame.dispose = gif::DisposalMethod::Background;
                encoder
                    .write_frame(&frame)
                    .map_err(|source| TransitionError::GifEncoding { source })?;
                progress_inc!();
            }

            frame_start += chunk_len;
        }

        info!(
            backend = "shatter",
            adapter_fallback = used_fallback_adapter,
            requested_size = %request.size(),
            dimensions = %dimensions,
            "render complete"
        );

        Ok(RenderReceipt::new(
            request.output_path().to_path_buf(),
            dimensions,
            total_frames,
            request.fps().get(),
        ))
    }
}

fn request_adapter(instance: &wgpu::Instance) -> Result<(wgpu::Adapter, bool), TransitionError> {
    let preferred = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    if let Ok(adapter) = preferred {
        return Ok((adapter, false));
    }

    let fallback = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: true,
        compatible_surface: None,
    }));
    match fallback {
        Ok(adapter) => Ok((adapter, true)),
        Err(error) => Err(TransitionError::GpuUnavailable {
            reason: format!("request_adapter failed: {error}"),
        }),
    }
}

fn compute_chunk_frames(
    device: &wgpu::Device,
    total_frames: u16,
    bytes_per_frame: usize,
) -> Result<usize, TransitionError> {
    if bytes_per_frame == 0 {
        return Err(TransitionError::GpuFailure {
            reason: "bytes_per_frame must be non-zero".to_string(),
        });
    }

    let limits = device.limits();
    let by_storage = usize::try_from(limits.max_storage_buffer_binding_size).unwrap_or(usize::MAX)
        / bytes_per_frame;
    let by_workgroups = usize::try_from(limits.max_compute_workgroups_per_dimension).unwrap_or(1);
    let capped = MAX_CHUNK_FRAMES.min(usize::from(total_frames));
    let chunk_frames = capped.min(by_storage.max(1)).min(by_workgroups.max(1));

    if chunk_frames == 0 {
        return Err(TransitionError::GpuFailure {
            reason: "unable to allocate a non-zero GPU chunk size".to_string(),
        });
    }

    Ok(chunk_frames)
}

fn map_staging_chunk(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    chunk_bytes: u64,
) -> Result<Vec<u8>, TransitionError> {
    let mapped_range = align_to(chunk_bytes, wgpu::MAP_ALIGNMENT);
    let slice = staging_buffer.slice(0..mapped_range);
    let (sender, receiver) = std::sync::mpsc::channel();

    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device
        .poll(wgpu::PollType::wait())
        .map_err(|error| TransitionError::GpuFailure {
            reason: format!("device poll failed: {error}"),
        })?;

    receiver
        .recv()
        .map_err(|error| TransitionError::GpuFailure {
            reason: format!("map_async callback channel failed: {error}"),
        })?
        .map_err(|error| TransitionError::GpuFailure {
            reason: format!("map_async failed: {error}"),
        })?;

    let mapped = slice.get_mapped_range();
    let bytes = mapped
        .get(
            0..usize::try_from(chunk_bytes).map_err(|_error| TransitionError::GpuFailure {
                reason: "chunk_bytes conversion failed".to_string(),
            })?,
        )
        .ok_or_else(|| TransitionError::GpuFailure {
            reason: "mapped slice shorter than expected chunk bytes".to_string(),
        })?
        .to_vec();
    drop(mapped);
    staging_buffer.unmap();

    Ok(bytes)
}

fn pack_rgba_to_u32(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn pack_rgba_to_u32_preserves_rgba_byte_order() {
        let packed = pack_rgba_to_u32(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(vec![0x0403_0201, 0x0807_0605], packed);
    }

    #[test]
    fn align_to_rounds_up() {
        assert_eq!(256, align_to(255, 256));
        assert_eq!(512, align_to(512, 256));
    }
}
