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
    /// GPU initialisation is unavailable on this host.
    #[error("gpu backend is unavailable: {reason}")]
    GpuUnavailable {
        /// Human-readable reason.
        reason: String,
    },
    /// GPU render path failed.
    #[error("gpu render failed: {reason}")]
    GpuFailure {
        /// Human-readable reason.
        reason: String,
    },
    /// GIF encoding failed.
    #[error("failed to encode transition gif")]
    GifEncoding {
        /// Underlying GIF encoding failure.
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
