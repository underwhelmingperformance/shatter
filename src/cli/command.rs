use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use std::num::NonZeroU16;
use std::path::PathBuf;
use tracing_subscriber::filter::LevelFilter;

use crate::RenderSize;
use crate::cli::render::RenderArgs;

/// Command-line options for the image shatter tool.
#[derive(Debug, Parser)]
#[command(
    name = "shatter",
    about = "Create an animated GIF that transitions between two images."
)]
pub struct Args {
    /// Override telemetry log verbosity.
    #[arg(long, global = true, value_enum)]
    log_level: Option<LogLevel>,
    /// Output format for command results. Defaults to `pretty` when stdout is a
    /// terminal, and `json` otherwise.
    #[arg(long, global = true, value_enum)]
    output_format: Option<OutputFormat>,
    /// Path to the source image at the start of the animation.
    #[arg(value_name = "FROM")]
    from: Option<PathBuf>,
    /// Path to the target image revealed by the transition.
    #[arg(value_name = "TO")]
    to: Option<PathBuf>,
    /// Output path for the rendered GIF.
    #[arg(value_name = "OUTPUT")]
    output: Option<PathBuf>,
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
    #[command(subcommand)]
    command: Option<Command>,
}

impl Args {
    /// Creates arguments directly without CLI parsing.
    ///
    /// ```
    /// use shatter::{Args, Command, RenderArgs};
    ///
    /// let args = Args::new(Command::Render(RenderArgs::default()));
    /// assert!(args.output_format().is_none());
    /// ```
    #[must_use]
    pub fn new(command: Command) -> Self {
        Self {
            log_level: None,
            output_format: None,
            from: None,
            to: None,
            output: None,
            size: RenderSize::Auto,
            frames: NonZeroU16::new(24).expect("24 is non-zero"),
            fps: NonZeroU16::new(16).expect("16 is non-zero"),
            seed: 0,
            command: Some(command),
        }
    }

    /// Returns an optional telemetry log-level override.
    ///
    /// ```
    /// use shatter::{Args, Command, RenderArgs};
    ///
    /// let args = Args::new(Command::Render(RenderArgs::default()));
    /// assert!(args.log_level().is_none());
    /// ```
    #[must_use]
    pub fn log_level(&self) -> Option<LogLevel> {
        self.log_level
    }

    /// Returns the explicitly selected output format, if any.
    ///
    /// ```
    /// use shatter::{Args, Command, RenderArgs};
    ///
    /// let args = Args::new(Command::Render(RenderArgs::default()));
    /// assert!(args.output_format().is_none());
    /// ```
    #[must_use]
    pub fn output_format(&self) -> Option<OutputFormat> {
        self.output_format
    }

    /// Consumes parsed args and returns the selected command.
    ///
    /// ```
    /// use shatter::{Args, Command, RenderArgs};
    ///
    /// let args = Args::new(Command::Render(RenderArgs::default()));
    /// assert!(matches!(args.into_command()?, Command::Render(_)));
    /// # Ok::<(), clap::Error>(())
    /// ```
    pub fn into_command(self) -> Result<Command, clap::Error> {
        if let Some(command) = self.command {
            return Ok(command);
        }

        let mut missing = Vec::new();
        if self.from.is_none() {
            missing.push("<FROM>");
        }
        if self.to.is_none() {
            missing.push("<TO>");
        }
        if self.output.is_none() {
            missing.push("<OUTPUT>");
        }

        if !missing.is_empty() {
            let mut cmd = <Self as CommandFactory>::command();
            let missing_flags = missing
                .iter()
                .map(|flag| format!("  {flag}"))
                .collect::<Vec<_>>()
                .join("\n");
            let message =
                format!("the following required arguments were not provided:\n{missing_flags}");
            return Err(cmd.error(clap::error::ErrorKind::MissingRequiredArgument, message));
        }

        Ok(Command::Render(RenderArgs::new(
            self.from.expect("validated above"),
            self.to.expect("validated above"),
            self.output.expect("validated above"),
            self.size,
            self.frames,
            self.fps,
            self.seed,
        )))
    }
}

/// Output format for command results.
#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable output.
    Pretty,
    /// Machine-readable JSON output.
    Json,
}

/// Log verbosity override for tracing events.
#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum LogLevel {
    /// Error-level events only.
    Error,
    /// Warning and error events.
    Warn,
    /// Informational, warning, and error events.
    Info,
    /// Debug and above.
    Debug,
    /// Full trace verbosity.
    Trace,
}

impl LogLevel {
    #[must_use]
    pub(crate) fn as_level_filter(self) -> LevelFilter {
        match self {
            Self::Error => LevelFilter::ERROR,
            Self::Warn => LevelFilter::WARN,
            Self::Info => LevelFilter::INFO,
            Self::Debug => LevelFilter::DEBUG,
            Self::Trace => LevelFilter::TRACE,
        }
    }
}

/// Supported CLI commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Render a pixel-shatter transition between two input images.
    Render(RenderArgs),
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn parse_rejects_invalid_size_value() {
        let result = Args::try_parse_from([
            "shatter", "render", "one.png", "two.png", "out.gif", "--size", "broken",
        ]);

        let error = result.expect_err("invalid size should fail argument parsing");
        assert_eq!(ErrorKind::ValueValidation, error.kind());
    }

    #[test]
    fn parse_defaults_render_size_to_auto() -> anyhow::Result<()> {
        let args = Args::try_parse_from(["shatter", "one.png", "two.png", "out.gif"])?;
        let command = args.into_command()?;

        let Command::Render(render_args) = command;
        assert_eq!(RenderSize::Auto, render_args.to_request().size());
        Ok(())
    }

    #[test]
    fn parse_defaults_to_render_when_subcommand_omitted() -> anyhow::Result<()> {
        let args = Args::try_parse_from(["shatter", "one.png", "two.png", "out.gif"])?;

        let command = args.into_command()?;
        assert!(matches!(command, Command::Render(_)));
        Ok(())
    }

    #[test]
    fn parse_omitted_subcommand_requires_render_paths() {
        let args = Args::try_parse_from(["shatter"])
            .expect("omitting subcommand should still parse top-level flags");

        let error = args
            .into_command()
            .expect_err("missing render arguments should fail");
        assert_eq!(ErrorKind::MissingRequiredArgument, error.kind());
    }
}
