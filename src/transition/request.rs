use std::num::NonZeroU16;
use std::path::{Path, PathBuf};

use crate::{PanelDimensions, RenderSize};

/// Render parameters independent of the input/output file paths.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RenderParams {
    pub size: RenderSize,
    pub frame_count: NonZeroU16,
    pub fps: NonZeroU16,
    pub hold_frames: u16,
    pub seed: u64,
}

/// Input parameters for a transition render operation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransitionRequest {
    from_path: PathBuf,
    to_path: PathBuf,
    output_path: PathBuf,
    size: RenderSize,
    frame_count: NonZeroU16,
    fps: NonZeroU16,
    hold_frames: u16,
    seed: u64,
}

impl TransitionRequest {
    /// Creates a transition request from file paths and render parameters.
    ///
    /// ```
    /// use std::num::NonZeroU16;
    ///
    /// use shatter::{PanelDimensions, RenderParams, RenderSize, TransitionRequest};
    ///
    /// let request = TransitionRequest::new(
    ///     "from.png",
    ///     "to.png",
    ///     "out.gif",
    ///     RenderParams {
    ///         size: RenderSize::Fixed(
    ///             PanelDimensions::new(64, 64).expect("64x64 should be valid"),
    ///         ),
    ///         frame_count: NonZeroU16::new(24).expect("24 is non-zero"),
    ///         fps: NonZeroU16::new(16).expect("16 is non-zero"),
    ///         hold_frames: 2,
    ///         seed: 42,
    ///     },
    /// );
    /// assert_eq!(2, request.hold_frames());
    /// assert_eq!(42, request.seed());
    /// ```
    #[must_use]
    pub fn new(
        from_path: impl Into<PathBuf>,
        to_path: impl Into<PathBuf>,
        output_path: impl Into<PathBuf>,
        params: RenderParams,
    ) -> Self {
        Self {
            from_path: from_path.into(),
            to_path: to_path.into(),
            output_path: output_path.into(),
            size: params.size,
            frame_count: params.frame_count,
            fps: params.fps,
            hold_frames: params.hold_frames,
            seed: params.seed,
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

    /// Returns the number of fully-static hold frames at both start and end.
    #[must_use]
    pub fn hold_frames(&self) -> u16 {
        self.hold_frames
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
