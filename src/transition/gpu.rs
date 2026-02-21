use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use shatter_macros::progress;
use tracing::{info, warn};

use crate::PanelDimensions;
use crate::media::{ImagePreparationError, decode_raster, decode_svg, is_svg};

use super::service::fps_to_delay_centiseconds;
use super::{RenderReceipt, TransitionError, TransitionRequest};

const MAX_CHUNK_FRAMES: usize = 64;
const QUANTIZE_WORKGROUP_SIZE: u32 = 256;
const DEFAULT_GRID_X: u32 = 20;
const DEFAULT_GRID_Y: u32 = 20;
const VERTICES_PER_CELL: u32 = 6;

// Uniform 6x7x6 RGB cube for GIF quantization.
const R_LEVELS: u32 = 6;
const G_LEVELS: u32 = 7;
const B_LEVELS: u32 = 6;
const PALETTE_COLORS: usize = (R_LEVELS * G_LEVELS * B_LEVELS) as usize; // 252
const TRANSPARENT_INDEX: u8 = PALETTE_COLORS as u8; // 252

// Algorithm note:
// Each cell of the grid is rendered as a textured instanced quad via a
// vertex/fragment pipeline.  The vertex shader applies zoom and radial drift
// with a simple perspective projection; the fragment shader samples the source
// image texture.  A Depth32Float buffer provides correct occlusion: cells with
// more motion receive lower depth values and naturally occlude stationary cells.
// This replaces the previous inverse-lookup compute shader, eliminating
// convergence failures for large displacements.
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
    grid_x: u32,
    grid_y: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizeParams {
    pixel_count: u32,
    width: u32,
    index_offset: u32,
    _pad: u32,
}

struct DecodedImages {
    from_pixels: Vec<u8>,
    to_pixels: Vec<u8>,
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
    shatter_pipeline: wgpu::RenderPipeline,
    shatter_bind_group_layout: wgpu::BindGroupLayout,
    quantize_pipeline: wgpu::ComputePipeline,
    quantize_bind_group_layout: wgpu::BindGroupLayout,
    used_fallback_adapter: bool,
}

/// Per-render GPU resources whose sizes depend on the input images and
/// output dimensions. Two staging buffers enable overlapping GPU work
/// with CPU readback/encoding (double-buffering).
struct RenderBuffers {
    shatter_bind_group: wgpu::BindGroup,
    quantize_bind_group: wgpu::BindGroup,
    render_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    index_buffer: wgpu::Buffer,
    staging_buffers: [wgpu::Buffer; 2],
    shatter_params_buffer: wgpu::Buffer,
    quantize_params_buffer: wgpu::Buffer,
    chunk_frames: usize,
    /// Bytes per frame for the index buffer (ceil(pixel_count/4) * 4).
    index_bytes_per_frame: usize,
    shatter_param_stride: u32,
    quantize_param_stride: u32,
}

struct RenderGifOptions {
    total_frames: u16,
    fps: u16,
    hold_frames: u16,
    seed: u64,
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
        let mut guard = self
            .gpu
            .lock()
            .expect("GPU context lock not poisoned");
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
        let mut output_file =
            std::fs::File::create(request.output_path()).map_err(|source| {
                TransitionError::OutputIo {
                    path: request.output_path().to_path_buf(),
                    source,
                }
            })?;
        let frames_total = usize::from(request.frame_count().get());
        progress_set_length!(frames_total);
        self.render_gif(
            &images,
            &RenderGifOptions {
                total_frames: request.frame_count().get(),
                fps: request.fps().get(),
                hold_frames: request.hold_frames(),
                seed: request.seed(),
            },
            &mut output_file,
            &mut |n| {
                progress_inc!(n);
            },
        )?;
        Ok(RenderReceipt::new(
            request.output_path().to_path_buf(),
            images.dimensions,
            request.frame_count().get(),
            request.fps().get(),
        ))
    }

    fn render_gif<W: Write>(
        &self,
        images: &DecodedImages,
        options: &RenderGifOptions,
        writer: W,
        on_progress: &mut dyn FnMut(usize),
    ) -> Result<(), TransitionError> {
        let ctx = self.ensure_gpu_context()?;
        let buffers = create_render_buffers(&ctx, images, options.total_frames)?;

        let out_width = images.dimensions.width();
        let out_height = images.dimensions.height();
        let total_frames = options.total_frames;
        let pixel_count = usize::from(out_width) * usize::from(out_height);

        let delay = fps_to_delay_centiseconds(options.fps);
        let frames_total = usize::from(total_frames);
        let words_per_frame = pixel_count.div_ceil(4);
        let mut chunk_encoder =
            ChunkEncoder::new(&ctx.device, writer, out_width, out_height, delay, pixel_count)?;
        let quantize_workgroups_x =
            (words_per_frame as u32).div_ceil(QUANTIZE_WORKGROUP_SIZE);
        let grid_x = DEFAULT_GRID_X;
        let grid_y = DEFAULT_GRID_Y;
        let instance_count = grid_x * grid_y;

        // Double-buffered dispatch loop: submit chunk N to
        // staging[N%2], then read back chunk N-1 from the other
        // staging buffer while the GPU is busy with chunk N.
        let mut frame_start = 0usize;
        let mut staging_index = 0usize;
        // (staging_idx, chunk_len, chunk_bytes)
        let mut pending: Option<(usize, usize, u64)> = None;

        while frame_start < frames_total {
            let chunk_len = (frames_total - frame_start).min(buffers.chunk_frames);

            // Pre-write all frame params for this chunk into the
            // dynamic-offset uniform buffers.
            let shatter_stride = buffers.shatter_param_stride as usize;
            let quantize_stride = buffers.quantize_param_stride as usize;
            let mut shatter_data = vec![0u8; shatter_stride * chunk_len];
            let mut quantize_data = vec![0u8; quantize_stride * chunk_len];

            for local_frame in 0..chunk_len {
                let global_frame = frame_start + local_frame;

                let sp = ShatterParams {
                    out_width: u32::from(out_width),
                    out_height: u32::from(out_height),
                    from_width: u32::from(images.from_width),
                    from_height: u32::from(images.from_height),
                    to_width: u32::from(images.to_width),
                    to_height: u32::from(images.to_height),
                    total_frames: u32::from(total_frames),
                    frame_start: u32::try_from(global_frame)
                        .expect("frame index fits u32"),
                    chunk_frames: 0,
                    hold_frames: u32::from(options.hold_frames),
                    seed_lo: options.seed as u32,
                    seed_hi: (options.seed >> 32) as u32,
                    grid_x,
                    grid_y,
                    _pad: 0,
                };
                let offset = local_frame * shatter_stride;
                shatter_data[offset..offset + std::mem::size_of::<ShatterParams>()]
                    .copy_from_slice(bytemuck::bytes_of(&sp));

                let qp = QuantizeParams {
                    pixel_count: pixel_count as u32,
                    width: u32::from(out_width),
                    index_offset: (local_frame * words_per_frame) as u32,
                    _pad: 0,
                };
                let offset = local_frame * quantize_stride;
                quantize_data[offset..offset + std::mem::size_of::<QuantizeParams>()]
                    .copy_from_slice(bytemuck::bytes_of(&qp));
            }

            ctx.queue
                .write_buffer(&buffers.shatter_params_buffer, 0, &shatter_data);
            ctx.queue
                .write_buffer(&buffers.quantize_params_buffer, 0, &quantize_data);

            let mut command_encoder =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("shatter-command-encoder"),
                    });

            for local_frame in 0..chunk_len {
                let shatter_offset =
                    (local_frame as u32) * buffers.shatter_param_stride;
                let quantize_offset =
                    (local_frame as u32) * buffers.quantize_param_stride;

                // Render pass: draw instanced quads.
                {
                    let mut render_pass =
                        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("shatter-render-pass"),
                            color_attachments: &[Some(
                                wgpu::RenderPassColorAttachment {
                                    view: &buffers.render_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.0,
                                            g: 0.0,
                                            b: 0.0,
                                            a: 0.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                },
                            )],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &buffers.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    render_pass.set_pipeline(&ctx.shatter_pipeline);
                    render_pass.set_bind_group(
                        0,
                        &buffers.shatter_bind_group,
                        &[shatter_offset],
                    );
                    render_pass.draw(0..VERTICES_PER_CELL, 0..instance_count);
                }

                // Quantize pass: render texture -> index_buffer slice.
                {
                    let mut pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("quantize-compute-pass"),
                            timestamp_writes: None,
                        });
                    pass.set_pipeline(&ctx.quantize_pipeline);
                    pass.set_bind_group(
                        0,
                        &buffers.quantize_bind_group,
                        &[quantize_offset],
                    );
                    pass.dispatch_workgroups(quantize_workgroups_x, 1, 1);
                }
            }

            let chunk_bytes = u64::try_from(chunk_len * buffers.index_bytes_per_frame)
                .expect("chunk byte size fits u64");
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
                chunk_encoder.encode_chunk(
                    &buffers.staging_buffers[prev_staging],
                    prev_len,
                    prev_bytes,
                )?;
                on_progress(prev_len);
            }

            pending = Some((staging_index, chunk_len, chunk_bytes));
            staging_index = 1 - staging_index;
            frame_start += chunk_len;
        }

        // Flush the final pending chunk.
        if let Some((prev_staging, prev_len, prev_bytes)) = pending.take() {
            chunk_encoder.encode_chunk(
                &buffers.staging_buffers[prev_staging],
                prev_len,
                prev_bytes,
            )?;
            on_progress(prev_len);
        }

        info!(
            backend = "shatter",
            adapter_fallback = ctx.used_fallback_adapter,
            dimensions = %images.dimensions,
            "render complete"
        );

        Ok(())
    }
}

/// Bundles the per-render GIF encoding state so that individual chunk
/// readbacks only need the varying staging-buffer parameters.
struct ChunkEncoder<'a, W: Write> {
    device: &'a wgpu::Device,
    encoder: gif::Encoder<W>,
    palette: Vec<u8>,
    pixel_count: usize,
    words_per_frame: usize,
    out_width: u16,
    out_height: u16,
    delay: u16,
}

impl<'a, W: Write> ChunkEncoder<'a, W> {
    fn new(
        device: &'a wgpu::Device,
        writer: W,
        out_width: u16,
        out_height: u16,
        delay: u16,
        pixel_count: usize,
    ) -> Result<Self, TransitionError> {
        let mut encoder = gif::Encoder::new(writer, out_width, out_height, &[])
            ?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            ?;
        Ok(Self {
            device,
            encoder,
            palette: build_palette(),
            pixel_count,
            words_per_frame: pixel_count.div_ceil(4),
            out_width,
            out_height,
            delay,
        })
    }

    fn encode_chunk(
        &mut self,
        staging_buffer: &wgpu::Buffer,
        chunk_len: usize,
        chunk_bytes: u64,
    ) -> Result<(), TransitionError> {
        let mapped_bytes = map_staging_chunk(self.device, staging_buffer, chunk_bytes)?;
        let bytes_per_frame = self.words_per_frame * std::mem::size_of::<u32>();
        for local_index in 0..chunk_len {
            let byte_start = local_index * bytes_per_frame;
            let byte_end = byte_start + bytes_per_frame;
            let packed = &mapped_bytes[byte_start..byte_end];
            // Each u32 word contains 4 packed indices (one per byte).
            // Truncate to exactly pixel_count indices.
            let indices: Vec<u8> = packed.iter().copied().take(self.pixel_count).collect();

            let mut frame = gif::Frame::from_palette_pixels(
                self.out_width,
                self.out_height,
                indices,
                self.palette.as_slice(),
                Some(TRANSPARENT_INDEX),
            );
            frame.delay = self.delay;
            frame.dispose = gif::DisposalMethod::Background;
            self.encoder
                .write_frame(&frame)
                ?;
        }
        Ok(())
    }
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

fn load_image(
    path: &Path,
    fit_to: Option<PanelDimensions>,
) -> Result<image::RgbaImage, ImagePreparationError> {
    let bytes = std::fs::read(path).map_err(ImagePreparationError::Read)?;
    if is_svg(path, &bytes) {
        decode_svg(&bytes, fit_to)
    } else {
        decode_raster(&bytes)
    }
}

fn load_images(request: &TransitionRequest) -> Result<DecodedImages, TransitionError> {
    let fit_to = match request.size() {
        crate::RenderSize::Fixed(dimensions) => Some(dimensions),
        crate::RenderSize::Auto => None,
    };

    let from_image = load_image(request.from_path(), fit_to)
        .map_err(|source| TransitionError::SourceImage {
            path: request.from_path().to_path_buf(),
            source,
        })?;
    let to_image =
        load_image(request.to_path(), fit_to).map_err(|source| TransitionError::SourceImage {
            path: request.to_path().to_path_buf(),
            source,
        })?;

    let from_width = try_image_dimension_u16(request.from_path(), "width", from_image.width())?;
    let from_height = try_image_dimension_u16(request.from_path(), "height", from_image.height())?;
    let to_width = try_image_dimension_u16(request.to_path(), "width", to_image.width())?;
    let to_height = try_image_dimension_u16(request.to_path(), "height", to_image.height())?;

    let from_dimensions = try_panel_dimensions(request.from_path(), from_width, from_height)?;
    let to_dimensions = try_panel_dimensions(request.to_path(), to_width, to_height)?;
    let dimensions = request.size().resolve(from_dimensions, to_dimensions);

    let from_pixels = from_image.into_raw();
    let to_pixels = to_image.into_raw();

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

fn try_image_dimension_u16(
    path: &Path,
    axis: &'static str,
    value: u32,
) -> Result<u16, TransitionError> {
    u16::try_from(value).map_err(|_source| TransitionError::InputImageDimensionTooLarge {
        path: path.to_path_buf(),
        axis,
        actual: value,
        max: u32::from(u16::MAX),
    })
}

fn try_panel_dimensions(
    path: &Path,
    width: u16,
    height: u16,
) -> Result<PanelDimensions, TransitionError> {
    PanelDimensions::new(width, height).ok_or(TransitionError::InvalidImageDimensions {
        path: path.to_path_buf(),
        width: u32::from(width),
        height: u32::from(height),
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
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
?;

    // --- Shatter render pipeline ---
    let shatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shatter-bind-group-layout"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(2, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
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
    let shatter_pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shatter-pipeline"),
            layout: Some(&shatter_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shatter_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shatter_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // --- Quantize pipeline ---
    let quantize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quantize-bind-group-layout"),
            entries: &[
                texture_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                uniform_entry(2, wgpu::ShaderStages::COMPUTE, true),
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

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_rgba_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    })
}

fn create_render_buffers(
    ctx: &GpuContext,
    images: &DecodedImages,
    total_frames: u16,
) -> Result<RenderBuffers, TransitionError> {
    let out_width = images.dimensions.width();
    let out_height = images.dimensions.height();

    let max_dim = ctx.device.limits().max_texture_dimension_2d;
    let all_widths = [images.from_width, images.to_width, out_width];
    let all_heights = [images.from_height, images.to_height, out_height];
    let max_used = all_widths
        .into_iter()
        .chain(all_heights)
        .map(u32::from)
        .max()
        .unwrap_or(0);
    if max_used > max_dim {
        return Err(TransitionError::ImageTooLarge {
            actual: max_used,
            max: max_dim,
        });
    }

    let pixel_count = usize::from(out_width) * usize::from(out_height);

    // Index buffer: each pixel -> 1 byte, packed 4 per u32.
    let words_per_frame = pixel_count.div_ceil(4);
    let index_bytes_per_frame = words_per_frame * std::mem::size_of::<u32>();
    let quantize_workgroups_x = (words_per_frame as u32).div_ceil(QUANTIZE_WORKGROUP_SIZE);

    let chunk_frames = compute_chunk_frames(
        &ctx.device,
        total_frames,
        index_bytes_per_frame,
        quantize_workgroups_x,
    )?;

    // Source image textures
    let from_texture = create_rgba_texture(
        &ctx.device,
        "from-texture",
        u32::from(images.from_width),
        u32::from(images.from_height),
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &from_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &images.from_pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * u32::from(images.from_width)),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: u32::from(images.from_width),
            height: u32::from(images.from_height),
            depth_or_array_layers: 1,
        },
    );
    let from_view = from_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let to_texture = create_rgba_texture(
        &ctx.device,
        "to-texture",
        u32::from(images.to_width),
        u32::from(images.to_height),
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &to_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &images.to_pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * u32::from(images.to_width)),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: u32::from(images.to_width),
            height: u32::from(images.to_height),
            depth_or_array_layers: 1,
        },
    );
    let to_view = to_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Render target
    let render_texture = create_rgba_texture(
        &ctx.device,
        "render-texture",
        u32::from(out_width),
        u32::from(out_height),
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Depth buffer
    let depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: u32::from(out_width),
            height: u32::from(out_height),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Index buffer and staging
    let index_buffer_size =
        u64::try_from(chunk_frames * index_bytes_per_frame).expect("index buffer size fits u64");
    let staging_buffer_size = align_to(index_buffer_size, wgpu::MAP_ALIGNMENT);

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

    // Dynamic-offset uniform buffers
    let min_align = u64::from(ctx.device.limits().min_uniform_buffer_offset_alignment);
    let shatter_param_stride =
        align_to(std::mem::size_of::<ShatterParams>() as u64, min_align) as u32;
    let quantize_param_stride =
        align_to(std::mem::size_of::<QuantizeParams>() as u64, min_align) as u32;

    let shatter_params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shatter-params-buffer"),
        size: u64::from(shatter_param_stride) * chunk_frames as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let quantize_params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantize-params-buffer"),
        size: u64::from(quantize_param_stride) * chunk_frames as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind groups
    let shatter_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shatter-bind-group"),
        layout: &ctx.shatter_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&from_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&to_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shatter_params_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShatterParams>() as u64,
                    ),
                }),
            },
        ],
    });

    let quantize_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quantize-bind-group"),
        layout: &ctx.quantize_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&render_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &quantize_params_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<QuantizeParams>() as u64,
                    ),
                }),
            },
        ],
    });

    Ok(RenderBuffers {
        shatter_bind_group,
        quantize_bind_group,
        render_view,
        depth_view,
        index_buffer,
        staging_buffers,
        shatter_params_buffer,
        quantize_params_buffer,
        chunk_frames,
        index_bytes_per_frame,
        shatter_param_stride,
        quantize_param_stride,
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
    let adapter = fallback?;
    Ok((adapter, true))
}

fn compute_chunk_frames(
    device: &wgpu::Device,
    total_frames: u16,
    index_bytes_per_frame: usize,
    words_per_frame_wg: u32,
) -> Result<usize, TransitionError> {
    assert!(index_bytes_per_frame > 0, "image has non-zero pixel area");

    let limits = device.limits();

    if words_per_frame_wg > limits.max_compute_workgroups_per_dimension {
        return Err(TransitionError::WorkgroupLimitExceeded {
            required: words_per_frame_wg,
            max: limits.max_compute_workgroups_per_dimension,
        });
    }

    let by_storage = usize::try_from(limits.max_storage_buffer_binding_size).unwrap_or(usize::MAX)
        / index_bytes_per_frame;
    let capped = MAX_CHUNK_FRAMES.min(usize::from(total_frames));
    let chunk_frames = capped.min(by_storage.max(1));

    // `by_storage` is at least 1 (guarded by `.max(1)`) and `total_frames`
    // is NonZeroU16, so `capped >= 1` and `chunk_frames >= 1`.
    assert!(chunk_frames > 0, "chunk_frames is at least 1");

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
        .map_err(|error| TransitionError::GpuRuntime {
            source: Box::new(error),
        })?;

    receiver
        .recv()
        .map_err(|error| TransitionError::GpuRuntime {
            source: Box::new(error),
        })?
        .map_err(|error| TransitionError::GpuRuntime {
            source: Box::new(error),
        })?;

    let mapped = slice.get_mapped_range();
    let chunk_len = usize::try_from(chunk_bytes).expect("chunk_bytes fits usize");
    let bytes = mapped[..chunk_len].to_vec();
    drop(mapped);
    staging_buffer.unmap();

    Ok(bytes)
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

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

    #[test]
    fn render_produces_distinct_frames_across_chunk() -> Result<(), Box<dyn std::error::Error>> {
        let (width, height): (u16, u16) = (32, 32);
        let pixel_count = usize::from(width) * usize::from(height);

        let mut from_pixels = vec![0u8; pixel_count * 4];
        let mut to_pixels = vec![0u8; pixel_count * 4];
        for y in 0..usize::from(height) {
            for x in 0..usize::from(width) {
                let i = (y * usize::from(width) + x) * 4;
                from_pixels[i] = (x * 8) as u8;
                from_pixels[i + 1] = (y * 8) as u8;
                from_pixels[i + 2] = 128;
                from_pixels[i + 3] = 255;

                to_pixels[i] = (255 - x * 8) as u8;
                to_pixels[i + 1] = (255 - y * 8) as u8;
                to_pixels[i + 2] = 64;
                to_pixels[i + 3] = 255;
            }
        }

        let dimensions =
            PanelDimensions::new(width, height).expect("32x32 dimensions are non-zero");
        let images = DecodedImages {
            from_pixels,
            to_pixels,
            from_width: width,
            from_height: height,
            to_width: width,
            to_height: height,
            dimensions,
        };

        let options = RenderGifOptions {
            total_frames: 4,
            fps: 10,
            hold_frames: 0,
            seed: 42,
        };

        let renderer = ShatterTransitionRenderer::new();
        let mut gif_buf = Vec::new();
        let result = renderer.render_gif(&images, &options, &mut gif_buf, &mut |_| {});
        assert_matches!(result, Ok(()) | Err(TransitionError::GpuUnavailable { .. }));
        if result.is_err() {
            return Ok(());
        }

        // Verify chunk size contract: 32x32 with 4 frames fits in a single chunk.
        let index_bytes_per_frame = pixel_count.div_ceil(4) * std::mem::size_of::<u32>();
        let quantize_workgroups_x =
            (pixel_count.div_ceil(4) as u32).div_ceil(QUANTIZE_WORKGROUP_SIZE);
        let ctx = renderer.ensure_gpu_context()?;
        let chunk_frames = compute_chunk_frames(
            &ctx.device,
            options.total_frames,
            index_bytes_per_frame,
            quantize_workgroups_x,
        )?;
        assert_eq!(4, chunk_frames);

        // Decode and hash each frame.
        let mut decoder = gif::DecodeOptions::new();
        decoder.set_color_output(gif::ColorOutput::Indexed);
        let mut reader = decoder.read_info(&gif_buf[..])?;
        let mut hashes = Vec::new();
        while let Some(frame) = reader.read_next_frame()? {
            let mut hasher = DefaultHasher::new();
            frame.buffer.hash(&mut hasher);
            hashes.push(hasher.finish());
        }

        let diffs: Vec<_> = hashes.windows(2).map(|w| w[0] != w[1]).collect();
        assert_eq!(vec![true, true, true], diffs);
        Ok(())
    }
}
