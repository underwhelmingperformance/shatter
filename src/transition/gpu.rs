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
const SHATTER_SHADER: &str = include_str!("shatter.wgsl");

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
            size: std::mem::size_of::<ShatterParams>() as u64,
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

            let params = ShatterParams {
                out_width: u32::from(out_width),
                out_height: u32::from(out_height),
                from_width: u32::from(from_width),
                from_height: u32::from(from_height),
                to_width: u32::from(to_width),
                to_height: u32::from(to_height),
                total_frames: u32::from(total_frames),
                frame_start: u32::try_from(frame_start).map_err(|_error| {
                    TransitionError::GpuFailure {
                        reason: "frame start overflow".to_string(),
                    }
                })?,
                chunk_frames: u32::try_from(chunk_len).map_err(|_error| {
                    TransitionError::GpuFailure {
                        reason: "chunk frame count overflow".to_string(),
                    }
                })?,
                hold_frames: u32::from(request.hold_frames()),
                seed_lo: request.seed() as u32,
                seed_hi: (request.seed() >> 32) as u32,
                _pad: [0; 3],
            };
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

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
                    u32::from(out_width).div_ceil(WORKGROUP_SIZE_X),
                    u32::from(out_height).div_ceil(WORKGROUP_SIZE_Y),
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
            let bytes_per_pixel = std::mem::size_of::<u32>();
            for local_index in 0..chunk_len {
                let byte_start = local_index * pixel_count * bytes_per_pixel;
                let byte_end = byte_start + pixel_count * bytes_per_pixel;
                let mut rgba = mapped_bytes[byte_start..byte_end].to_vec();

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
