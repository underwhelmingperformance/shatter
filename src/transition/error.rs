use std::path::PathBuf;

use thiserror::Error;

use crate::media::ImagePreparationError;

/// Errors returned while rendering a transition GIF.
#[derive(Debug, Error)]
pub enum TransitionError {
    /// Source image normalisation failed.
    #[error("failed to prepare source image `{path}`")]
    SourceImage {
        /// Source image path.
        path: PathBuf,
        /// Underlying decode or transformation failure.
        source: ImagePreparationError,
    },
    /// Input image dimensions exceed the renderer's internal coordinate range.
    #[error(
        "input image `{path}` {axis} {actual} exceeds supported limit of {max}; \
         use --size to downscale"
    )]
    InputImageDimensionTooLarge {
        /// Input image path.
        path: PathBuf,
        /// Offending dimension axis.
        axis: &'static str,
        /// Observed dimension value.
        actual: u32,
        /// Supported maximum for this dimension.
        max: u32,
    },
    /// Input image reports invalid zero dimensions.
    #[error("input image `{path}` has invalid zero dimensions ({width}x{height})")]
    InvalidImageDimensions {
        /// Input image path.
        path: PathBuf,
        /// Decoded image width.
        width: u32,
        /// Decoded image height.
        height: u32,
    },
    /// No GPU adapter could be acquired.
    #[error("no GPU adapter available")]
    GpuUnavailable {
        #[from]
        source: wgpu::RequestAdapterError,
    },
    /// GPU device creation failed.
    #[error("failed to initialise GPU device")]
    DeviceInit {
        #[from]
        source: wgpu::RequestDeviceError,
    },
    /// Input image dimensions exceed what the GPU device supports.
    #[error("image dimension {actual} exceeds GPU limit of {max}; use --size to downscale")]
    ImageTooLarge {
        /// Largest dimension across all input/output textures.
        actual: u32,
        /// Maximum texture dimension supported by the device.
        max: u32,
    },
    /// Image requires more compute workgroups than the device supports.
    #[error(
        "image requires {required} compute workgroups but device supports {max}; \
         use --size to downscale"
    )]
    WorkgroupLimitExceeded {
        /// Workgroups needed for this image.
        required: u32,
        /// Device workgroup-per-dimension limit.
        max: u32,
    },
    /// A GPU runtime operation (buffer mapping, device poll) failed.
    #[error("GPU operation failed")]
    GpuRuntime {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// GIF encoding failed.
    #[error("failed to encode transition gif")]
    GifEncoding {
        #[from]
        source: gif::EncodingError,
    },
    /// Output file write failed.
    #[error("failed to write transition output `{path}`")]
    OutputIo {
        /// Output path.
        path: PathBuf,
        /// Underlying write failure.
        source: std::io::Error,
    },
}
