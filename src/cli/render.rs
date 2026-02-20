use std::io;
use std::num::NonZeroU16;
use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use serde::Serialize;
use tracing::instrument;

use crate::cli::OutputFormat;
use crate::{RenderReceipt, RenderSize, TransitionRequest, TransitionService};

pub(crate) const DEFAULT_DURATION_SECONDS: f32 = 1.5;
pub(crate) const DEFAULT_HOLD_SECONDS: f32 = 0.125;

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
    /// Total animation duration in seconds.
    #[arg(long, default_value_t = DEFAULT_DURATION_SECONDS, value_parser = parse_positive_seconds)]
    duration_seconds: f32,
    /// Playback frame rate.
    #[arg(long, default_value_t = NonZeroU16::new(16).expect("16 is non-zero"))]
    fps: NonZeroU16,
    /// Additional static hold time, in seconds, at both the start and end.
    #[arg(long, default_value_t = DEFAULT_HOLD_SECONDS, value_parser = parse_non_negative_seconds)]
    hold_seconds: f32,
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
            duration_seconds: DEFAULT_DURATION_SECONDS,
            fps: NonZeroU16::new(16).expect("16 is non-zero"),
            hold_seconds: DEFAULT_HOLD_SECONDS,
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
        duration_seconds: f32,
        fps: NonZeroU16,
        hold_seconds: f32,
        seed: u64,
    ) -> Self {
        Self {
            from,
            to,
            output,
            size,
            duration_seconds,
            fps,
            hold_seconds,
            seed,
        }
    }

    pub(crate) fn to_request(&self) -> TransitionRequest {
        let frame_count = duration_seconds_to_frame_count(self.duration_seconds, self.fps);
        let hold_frames = hold_seconds_to_frame_count(self.hold_seconds, self.fps);
        TransitionRequest::new(
            self.from.clone(),
            self.to.clone(),
            self.output.clone(),
            self.size,
            frame_count,
            self.fps,
            hold_frames,
            self.seed,
        )
    }
}

pub(crate) fn parse_positive_seconds(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_error| format!("`{raw}` is not a valid seconds value"))?;
    if !value.is_finite() || value <= 0.0 {
        return Err("duration must be a finite value greater than zero".to_string());
    }
    Ok(value)
}

pub(crate) fn parse_non_negative_seconds(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_error| format!("`{raw}` is not a valid seconds value"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(
            "hold duration must be a finite value greater than or equal to zero".to_string(),
        );
    }
    Ok(value)
}

fn duration_seconds_to_frame_count(duration_seconds: f32, fps: NonZeroU16) -> NonZeroU16 {
    let estimated = (duration_seconds * f32::from(fps.get())).round();
    let clamped = estimated.clamp(1.0, f32::from(u16::MAX));
    NonZeroU16::new(clamped as u16).expect("clamped frame count must be non-zero")
}

fn hold_seconds_to_frame_count(hold_seconds: f32, fps: NonZeroU16) -> u16 {
    let estimated = (hold_seconds * f32::from(fps.get())).round();
    let clamped = estimated.clamp(0.0, f32::from(u16::MAX));
    clamped as u16
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
        assert_eq!(2, request.hold_frames());
        assert_eq!(0, request.seed());
    }

    #[test]
    fn to_request_converts_seconds_to_frame_counts() {
        let args = RenderArgs::new(
            PathBuf::from("from.png"),
            PathBuf::from("to.png"),
            PathBuf::from("out.gif"),
            RenderSize::Auto,
            1.2,
            NonZeroU16::new(10).expect("10 is non-zero"),
            0.3,
            9,
        );
        let request = args.to_request();

        assert_eq!(12, request.frame_count().get());
        assert_eq!(3, request.hold_frames());
        assert_eq!(9, request.seed());
    }
}
