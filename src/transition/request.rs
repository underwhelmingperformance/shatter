use std::num::NonZeroU16;
use std::path::{Path, PathBuf};

use crate::{PanelDimensions, RenderSize};

/// Input parameters for a transition render operation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransitionRequest {
    from_path: PathBuf,
    to_path: PathBuf,
    output_path: PathBuf,
    size: RenderSize,
    frame_count: NonZeroU16,
    fps: NonZeroU16,
    seed: u64,
}

impl TransitionRequest {
    /// Creates a transition request.
    ///
    /// ```
    /// use std::num::NonZeroU16;
    ///
    /// use shatter::{PanelDimensions, RenderSize, TransitionRequest};
    ///
    /// let request = TransitionRequest::new(
    ///     "from.png",
    ///     "to.png",
    ///     "out.gif",
    ///     RenderSize::Fixed(PanelDimensions::new(64, 64).expect("64x64 should be valid")),
    ///     NonZeroU16::new(24).expect("24 is non-zero"),
    ///     NonZeroU16::new(16).expect("16 is non-zero"),
    ///     42,
    /// );
    /// assert_eq!(42, request.seed());
    /// ```
    #[must_use]
    pub fn new(
        from_path: impl Into<PathBuf>,
        to_path: impl Into<PathBuf>,
        output_path: impl Into<PathBuf>,
        size: RenderSize,
        frame_count: NonZeroU16,
        fps: NonZeroU16,
        seed: u64,
    ) -> Self {
        Self {
            from_path: from_path.into(),
            to_path: to_path.into(),
            output_path: output_path.into(),
            size,
            frame_count,
            fps,
            seed,
        }
    }

    /// Returns the source image path.
    #[must_use]
    pub fn from_path(&self) -> &Path {
        &self.from_path
    }

    /// Returns the destination image path.
    #[must_use]
    pub fn to_path(&self) -> &Path {
        &self.to_path
    }

    /// Returns output GIF path.
    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Returns output sizing strategy.
    #[must_use]
    pub fn size(&self) -> RenderSize {
        self.size
    }

    /// Returns frame count.
    #[must_use]
    pub fn frame_count(&self) -> NonZeroU16 {
        self.frame_count
    }

    /// Returns output frame rate.
    #[must_use]
    pub fn fps(&self) -> NonZeroU16 {
        self.fps
    }

    /// Returns the deterministic random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Successful transition render result.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RenderReceipt {
    output_path: PathBuf,
    dimensions: PanelDimensions,
    frame_count: u16,
    fps: u16,
}

impl RenderReceipt {
    /// Creates a render receipt.
    ///
    /// ```
    /// use shatter::{PanelDimensions, RenderReceipt};
    ///
    /// let receipt = RenderReceipt::new(
    ///     "out.gif".into(),
    ///     PanelDimensions::new(32, 32).expect("32x32 should be valid"),
    ///     24,
    ///     16,
    /// );
    /// assert_eq!(24, receipt.frame_count());
    /// ```
    #[must_use]
    pub fn new(
        output_path: PathBuf,
        dimensions: PanelDimensions,
        frame_count: u16,
        fps: u16,
    ) -> Self {
        Self {
            output_path,
            dimensions,
            frame_count,
            fps,
        }
    }

    /// Returns output GIF path.
    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Returns output dimensions.
    #[must_use]
    pub fn dimensions(&self) -> PanelDimensions {
        self.dimensions
    }

    /// Returns rendered frame count.
    #[must_use]
    pub fn frame_count(&self) -> u16 {
        self.frame_count
    }

    /// Returns playback frame rate.
    #[must_use]
    pub fn fps(&self) -> u16 {
        self.fps
    }
}
