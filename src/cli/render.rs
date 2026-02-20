use std::io;
use std::num::NonZeroU16;
use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use serde::Serialize;
use tracing::instrument;

use crate::cli::OutputFormat;
use crate::{RenderReceipt, RenderSize, TransitionRequest, TransitionService};

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RenderResult {
    Rendered {
        output: String,
        dimensions: String,
        frames: u16,
        fps: u16,
    },
}

/// Arguments for GIF transition rendering.
#[derive(Debug, Clone, Args)]
pub struct RenderArgs {
    /// Path to the source image at the start of the animation.
    #[arg(value_name = "FROM")]
    from: PathBuf,
    /// Path to the target image revealed by the transition.
    #[arg(value_name = "TO")]
    to: PathBuf,
    /// Output path for the rendered GIF.
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,
    /// Output size strategy: `auto` or `WIDTHxHEIGHT`.
    #[arg(long, default_value = "auto")]
    size: RenderSize,
    /// Number of frames to generate.
    #[arg(long, default_value_t = NonZeroU16::new(24).expect("24 is non-zero"))]
    frames: NonZeroU16,
    /// Playback frame rate.
    #[arg(long, default_value_t = NonZeroU16::new(16).expect("16 is non-zero"))]
    fps: NonZeroU16,
    /// Seed used to keep the pixel-order deterministic.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl Default for RenderArgs {
    fn default() -> Self {
        Self {
            from: PathBuf::from("from.png"),
            to: PathBuf::from("to.png"),
            output: PathBuf::from("out.gif"),
            size: RenderSize::Auto,
            frames: NonZeroU16::new(24).expect("24 is non-zero"),
            fps: NonZeroU16::new(16).expect("16 is non-zero"),
            seed: 0,
        }
    }
}

impl RenderArgs {
    pub(crate) fn new(
        from: PathBuf,
        to: PathBuf,
        output: PathBuf,
        size: RenderSize,
        frames: NonZeroU16,
        fps: NonZeroU16,
        seed: u64,
    ) -> Self {
        Self {
            from,
            to,
            output,
            size,
            frames,
            fps,
            seed,
        }
    }

    pub(crate) fn to_request(&self) -> TransitionRequest {
        TransitionRequest::new(
            self.from.clone(),
            self.to.clone(),
            self.output.clone(),
            self.size,
            self.frames,
            self.fps,
            self.seed,
        )
    }
}

/// Executes the top-level `render` command.
#[instrument(skip(service, args, out), level = "info", fields(?output_format))]
pub(crate) fn run<W>(
    service: &dyn TransitionService,
    args: &RenderArgs,
    out: &mut W,
    output_format: OutputFormat,
) -> Result<()>
where
    W: io::Write,
{
    let request = args.to_request();
    let receipt = service.render(request)?;

    match output_format {
        OutputFormat::Pretty => write_pretty(out, &receipt)?,
        OutputFormat::Json => write_json(out, &receipt)?,
    }

    Ok(())
}

fn write_pretty(out: &mut impl io::Write, receipt: &RenderReceipt) -> Result<()> {
    writeln!(
        out,
        "Rendered {} ({} @ {} fps) -> {}",
        receipt.frame_count(),
        receipt.dimensions(),
        receipt.fps(),
        receipt.output_path().display()
    )?;
    Ok(())
}

fn write_json(out: &mut impl io::Write, receipt: &RenderReceipt) -> Result<()> {
    let payload = RenderResult::Rendered {
        output: receipt.output_path().display().to_string(),
        dimensions: receipt.dimensions().to_string(),
        frames: receipt.frame_count(),
        fps: receipt.fps(),
    };
    serde_json::to_writer_pretty(&mut *out, &payload)?;
    writeln!(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn default_args_emit_expected_defaults() {
        let args = RenderArgs::default();
        let request = args.to_request();

        assert_eq!(RenderSize::Auto, request.size());
        assert_eq!(24, request.frame_count().get());
        assert_eq!(16, request.fps().get());
        assert_eq!(0, request.seed());
    }
}
