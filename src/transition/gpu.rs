use std::sync::{Arc, Mutex};

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
const QUANTIZE_WORKGROUP_SIZE: u32 = 256;

// Uniform 6x7x6 RGB cube for GIF quantization.
const R_LEVELS: u32 = 6;
const G_LEVELS: u32 = 7;
const B_LEVELS: u32 = 6;
const PALETTE_COLORS: usize = (R_LEVELS * G_LEVELS * B_LEVELS) as usize; // 252
const TRANSPARENT_INDEX: u8 = PALETTE_COLORS as u8; // 252

// Algorithm note:
// This shader simulates an explosion behind the image plane. All cell-sized
// debris flies forward towards the camera, scaling up via perspective projection.
// Cells further from centre drift off-screen faster (parallax), while centre
// cells loom large before fading. Edges crack first, the centre holds longest.
// Once the source image has fully dispersed, the process reverses to assemble
// the target image.
const SHATTER_SHADER: &str = include_str!("shatter.wgsl");
const QUANTIZE_SHADER: &str = include_str!("quantize.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShatterParams {
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
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizeParams {
    pixel_count: u32,
    chunk_frames: u32,
}

struct DecodedImages {
    from_pixels: Vec<u32>,
    to_pixels: Vec<u32>,
    from_width: u16,
    from_height: u16,
    to_width: u16,
    to_height: u16,
    dimensions: PanelDimensions,
}

/// Long-lived GPU state that is independent of any particular render
/// request: adapter, device, queue, compiled pipelines, and the bind
/// group layouts. Created lazily on the first render and reused for
/// all subsequent renders to avoid repeated device/shader init.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shatter_pipeline: wgpu::ComputePipeline,
    shatter_bind_group_layout: wgpu::BindGroupLayout,
    quantize_pipeline: wgpu::ComputePipeline,
    quantize_bind_group_layout: wgpu::BindGroupLayout,
    used_fallback_adapter: bool,
}

/// Per-render GPU resources whose sizes depend on the input images and
/// output dimensions. Two staging buffers enable overlapping GPU
/// compute with CPU readback/encoding (double-buffering).
struct RenderBuffers {
    shatter_bind_group: wgpu::BindGroup,
    quantize_bind_group: wgpu::BindGroup,
    index_buffer: wgpu::Buffer,
    staging_buffers: [wgpu::Buffer; 2],
    params_buffer: wgpu::Buffer,
    quantize_params_buffer: wgpu::Buffer,
    chunk_frames: usize,
    /// Bytes per frame for the index buffer (ceil(pixel_count/4) * 4).
    index_bytes_per_frame: usize,
}

pub(super) struct ShatterTransitionRenderer {
    gpu: Mutex<Option<Arc<GpuContext>>>,
}

impl std::fmt::Debug for ShatterTransitionRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let initialized = self
            .gpu
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        f.debug_struct("ShatterTransitionRenderer")
            .field("gpu_initialized", &initialized)
            .finish()
    }
}

impl Default for ShatterTransitionRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShatterTransitionRenderer {
    pub(super) fn new() -> Self {
        Self {
            gpu: Mutex::new(None),
        }
    }

    fn ensure_gpu_context(&self) -> Result<Arc<GpuContext>, TransitionError> {
        let mut guard = self.gpu.lock().map_err(|_error| {
            TransitionError::GpuFailure {
                reason: "GPU context lock poisoned".to_string(),
            }
        })?;
        if guard.is_none() {
            *guard = Some(Arc::new(init_gpu_context()?));
        }
        Ok(Arc::clone(guard.as_ref().unwrap()))
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
        let images = load_images(request)?;
        let ctx = self.ensure_gpu_context()?;
        let buffers = create_render_buffers(&ctx, &images, request)?;

        let out_width = images.dimensions.width();
        let out_height = images.dimensions.height();
        let total_frames = request.frame_count().get();
        let pixel_count = usize::from(out_width) * usize::from(out_height);

        let palette = build_palette();

        let mut output_file =
            std::fs::File::create(request.output_path()).map_err(|source| {
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
        let workgroups_x = u32::from(out_width).div_ceil(WORKGROUP_SIZE_X);
        let workgroups_y = u32::from(out_height).div_ceil(WORKGROUP_SIZE_Y);
        let words_per_frame = pixel_count.div_ceil(4);
        let quantize_workgroups_x =
            (words_per_frame as u32).div_ceil(QUANTIZE_WORKGROUP_SIZE);

        // Double-buffered dispatch loop: submit chunk N to
        // staging[N%2], then read back chunk N-1 from the other
        // staging buffer while the GPU is busy with chunk N.
        progress_set_length!(frames_total);
        let mut frame_start = 0usize;
        let mut staging_index = 0usize;
        // (staging_idx, chunk_len, chunk_bytes)
        let mut pending: Option<(usize, usize, u64)> = None;

        while frame_start < frames_total {
            let chunk_len = (frames_total - frame_start).min(buffers.chunk_frames);
            let chunk_len_u32 =
                u32::try_from(chunk_len).map_err(|_error| TransitionError::GpuFailure {
                    reason: "dispatch chunk overflow".to_string(),
                })?;

            let params = ShatterParams {
                out_width: u32::from(out_width),
                out_height: u32::from(out_height),
                from_width: u32::from(images.from_width),
                from_height: u32::from(images.from_height),
                to_width: u32::from(images.to_width),
                to_height: u32::from(images.to_height),
                total_frames: u32::from(total_frames),
                frame_start: u32::try_from(frame_start).map_err(|_error| {
                    TransitionError::GpuFailure {
                        reason: "frame start overflow".to_string(),
                    }
                })?,
                chunk_frames: chunk_len_u32,
                hold_frames: u32::from(request.hold_frames()),
                seed_lo: request.seed() as u32,
                seed_hi: (request.seed() >> 32) as u32,
                _pad: [0; 3],
            };
            ctx.queue
                .write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

            let quantize_params = QuantizeParams {
                pixel_count: pixel_count as u32,
                chunk_frames: chunk_len_u32,
            };
            ctx.queue.write_buffer(
                &buffers.quantize_params_buffer,
                0,
                bytemuck::bytes_of(&quantize_params),
            );

            let mut command_encoder =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("shatter-command-encoder"),
                    });

            // Pass 1: shatter effect (RGBA output).
            {
                let mut pass =
                    command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("shatter-compute-pass"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&ctx.shatter_pipeline);
                pass.set_bind_group(0, &buffers.shatter_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, chunk_len_u32);
            }

            // Pass 2: quantize RGBA to palette indices.
            {
                let mut pass =
                    command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("quantize-compute-pass"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&ctx.quantize_pipeline);
                pass.set_bind_group(0, &buffers.quantize_bind_group, &[]);
                pass.dispatch_workgroups(
                    quantize_workgroups_x,
                    1,
                    chunk_len_u32,
                );
            }

            let chunk_bytes =
                u64::try_from(chunk_len * buffers.index_bytes_per_frame).map_err(
                    |_error| TransitionError::GpuFailure {
                        reason: "chunk byte size overflow".to_string(),
                    },
                )?;
            command_encoder.copy_buffer_to_buffer(
                &buffers.index_buffer,
                0,
                &buffers.staging_buffers[staging_index],
                0,
                chunk_bytes,
            );
            ctx.queue.submit(Some(command_encoder.finish()));

            // While the GPU computes chunk N, read back chunk N-1.
            if let Some((prev_staging, prev_len, prev_bytes)) = pending.take() {
                encode_indexed_frames(
                    &ctx,
                    &buffers,
                    prev_staging,
                    prev_len,
                    prev_bytes,
                    pixel_count,
                    words_per_frame,
                    out_width,
                    out_height,
                    delay,
                    &palette,
                    &mut encoder,
                )?;
                progress_inc!(prev_len);
            }

            pending = Some((staging_index, chunk_len, chunk_bytes));
            staging_index = 1 - staging_index;
            frame_start += chunk_len;
        }

        // Flush the final pending chunk.
        if let Some((prev_staging, prev_len, prev_bytes)) = pending.take() {
            encode_indexed_frames(
                &ctx,
                &buffers,
                prev_staging,
                prev_len,
                prev_bytes,
                pixel_count,
                words_per_frame,
                out_width,
                out_height,
                delay,
                &palette,
                &mut encoder,
            )?;
            progress_inc!(prev_len);
        }

        info!(
            backend = "shatter",
            adapter_fallback = ctx.used_fallback_adapter,
            requested_size = %request.size(),
            dimensions = %images.dimensions,
            "render complete"
        );

        Ok(RenderReceipt::new(
            request.output_path().to_path_buf(),
            images.dimensions,
            total_frames,
            request.fps().get(),
        ))
    }
}

/// Read back index data from a staging buffer and write GIF frames.
fn encode_indexed_frames(
    ctx: &GpuContext,
    buffers: &RenderBuffers,
    staging_index: usize,
    chunk_len: usize,
    chunk_bytes: u64,
    pixel_count: usize,
    words_per_frame: usize,
    out_width: u16,
    out_height: u16,
    delay: u16,
    palette: &[u8],
    encoder: &mut gif::Encoder<&mut std::fs::File>,
) -> Result<(), TransitionError> {
    let mapped_bytes = map_staging_chunk(
        &ctx.device,
        &buffers.staging_buffers[staging_index],
        chunk_bytes,
    )?;
    let bytes_per_frame = words_per_frame * std::mem::size_of::<u32>();
    for local_index in 0..chunk_len {
        let byte_start = local_index * bytes_per_frame;
        let byte_end = byte_start + bytes_per_frame;
        let packed = &mapped_bytes[byte_start..byte_end];
        // Each u32 word contains 4 packed indices (one per byte).
        // Truncate to exactly pixel_count indices.
        let indices: Vec<u8> = packed.iter().copied().take(pixel_count).collect();

        let frame = gif::Frame::from_palette_pixels(
            out_width,
            out_height,
            indices,
            palette,
            Some(TRANSPARENT_INDEX),
        );
        let mut frame = frame;
        frame.delay = delay;
        frame.dispose = gif::DisposalMethod::Background;
        encoder
            .write_frame(&frame)
            .map_err(|source| TransitionError::GifEncoding { source })?;
    }
    Ok(())
}

/// Build the 6x7x6 uniform RGB palette (252 colours) plus one
/// transparent entry at index 252. Returns packed RGB triplets
/// suitable for `gif::Frame::from_palette_pixels`.
fn build_palette() -> Vec<u8> {
    let mut palette = Vec::with_capacity((PALETTE_COLORS + 1) * 3);
    for ri in 0..R_LEVELS {
        for gi in 0..G_LEVELS {
            for bi in 0..B_LEVELS {
                let r = ((ri * 255) / (R_LEVELS - 1)) as u8;
                let g = ((gi * 255) / (G_LEVELS - 1)) as u8;
                let b = ((bi * 255) / (B_LEVELS - 1)) as u8;
                palette.push(r);
                palette.push(g);
                palette.push(b);
            }
        }
    }
    // Transparent colour (value doesn't matter; alpha handles it).
    palette.push(0);
    palette.push(0);
    palette.push(0);
    palette
}

fn load_images(request: &TransitionRequest) -> Result<DecodedImages, TransitionError> {
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

    let from_pixels = pack_rgba_to_u32(from_image.as_raw());
    let to_pixels = pack_rgba_to_u32(to_image.as_raw());

    Ok(DecodedImages {
        from_pixels,
        to_pixels,
        from_width,
        from_height,
        to_width,
        to_height,
        dimensions,
    })
}

fn init_gpu_context() -> Result<GpuContext, TransitionError> {
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

    // --- Shatter pipeline ---
    let shatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shatter-bind-group-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let shatter_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shatter-pipeline-layout"),
            bind_group_layouts: &[&shatter_bind_group_layout],
            push_constant_ranges: &[],
        });
    let shatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shatter-shader"),
        source: wgpu::ShaderSource::Wgsl(SHATTER_SHADER.into()),
    });
    let shatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shatter-pipeline"),
        layout: Some(&shatter_pipeline_layout),
        module: &shatter_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // --- Quantize pipeline ---
    let quantize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quantize-bind-group-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });
    let quantize_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quantize-pipeline-layout"),
            bind_group_layouts: &[&quantize_bind_group_layout],
            push_constant_ranges: &[],
        });
    let quantize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quantize-shader"),
        source: wgpu::ShaderSource::Wgsl(QUANTIZE_SHADER.into()),
    });
    let quantize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("quantize-pipeline"),
        layout: Some(&quantize_pipeline_layout),
        module: &quantize_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(GpuContext {
        device,
        queue,
        shatter_pipeline,
        shatter_bind_group_layout,
        quantize_pipeline,
        quantize_bind_group_layout,
        used_fallback_adapter,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_render_buffers(
    ctx: &GpuContext,
    images: &DecodedImages,
    request: &TransitionRequest,
) -> Result<RenderBuffers, TransitionError> {
    let from_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("from-buffer"),
        contents: bytemuck::cast_slice(&images.from_pixels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let to_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("to-buffer"),
        contents: bytemuck::cast_slice(&images.to_pixels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let total_frames = request.frame_count().get();
    let out_width = images.dimensions.width();
    let out_height = images.dimensions.height();
    let pixel_count = usize::from(out_width) * usize::from(out_height);
    let rgba_bytes_per_frame = pixel_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| TransitionError::GpuFailure {
            reason: "frame byte size overflow".to_string(),
        })?;

    // Index buffer: each pixel → 1 byte, packed 4 per u32.
    let words_per_frame = pixel_count.div_ceil(4);
    let index_bytes_per_frame = words_per_frame * std::mem::size_of::<u32>();

    let chunk_frames =
        compute_chunk_frames(&ctx.device, total_frames, rgba_bytes_per_frame)?;

    let output_buffer_size =
        u64::try_from(chunk_frames * rgba_bytes_per_frame).map_err(|_error| {
            TransitionError::GpuFailure {
                reason: "output buffer size overflow".to_string(),
            }
        })?;
    let index_buffer_size =
        u64::try_from(chunk_frames * index_bytes_per_frame).map_err(|_error| {
            TransitionError::GpuFailure {
                reason: "index buffer size overflow".to_string(),
            }
        })?;
    let staging_buffer_size = align_to(index_buffer_size, wgpu::MAP_ALIGNMENT);

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output-buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let index_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("index-buffer"),
        size: index_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffers = [
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer-0"),
            size: staging_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }),
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer-1"),
            size: staging_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }),
    ];
    let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params-buffer"),
        size: std::mem::size_of::<ShatterParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let quantize_params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantize-params-buffer"),
        size: std::mem::size_of::<QuantizeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shatter_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shatter-bind-group"),
        layout: &ctx.shatter_bind_group_layout,
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

    let quantize_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quantize-bind-group"),
        layout: &ctx.quantize_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: quantize_params_buffer.as_entire_binding(),
            },
        ],
    });

    Ok(RenderBuffers {
        shatter_bind_group,
        quantize_bind_group,
        index_buffer,
        staging_buffers,
        params_buffer,
        quantize_params_buffer,
        chunk_frames,
        index_bytes_per_frame,
    })
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

    #[test]
    fn build_palette_has_correct_size_and_entries() {
        let palette = build_palette();
        // 252 colours + 1 transparent = 253 entries, 3 bytes each
        assert_eq!(253 * 3, palette.len());
        // First entry: R=0, G=0, B=0
        assert_eq!([0, 0, 0], &palette[0..3]);
        // Last non-transparent: R=255, G=255, B=255
        let last = (PALETTE_COLORS - 1) * 3;
        assert_eq!([255, 255, 255], &palette[last..last + 3]);
    }
}
