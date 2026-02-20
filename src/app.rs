use std::io;

use anyhow::Result;
use tracing::instrument;

use crate::cli::{Command, LogLevel, OutputFormat};
use crate::telemetry;
use crate::terminal::{SystemTerminalClient, TerminalClient};
use crate::transition::TransitionService;

/// Runs a command with default telemetry settings.
///
/// # Errors
///
/// Returns an error if telemetry initialisation fails, command execution fails,
/// or output writing fails.
pub async fn run<W>(
    command: Command,
    out: &mut W,
    transition_service: Box<dyn TransitionService>,
) -> Result<()>
where
    W: io::Write,
{
    run_with_log_level(command, out, transition_service, None, OutputFormat::Pretty).await
}

/// Runs a command with an explicit telemetry log-level override.
///
/// # Errors
///
/// Returns an error if telemetry initialisation fails, command execution fails,
/// or output writing fails.
pub async fn run_with_log_level<W>(
    command: Command,
    out: &mut W,
    transition_service: Box<dyn TransitionService>,
    log_level: Option<LogLevel>,
    output_format: OutputFormat,
) -> Result<()>
where
    W: io::Write,
{
    run_with_clients_and_log_level(
        command,
        out,
        &SystemTerminalClient,
        transition_service,
        log_level,
        output_format,
    )
    .await
}

/// Runs a command with injected terminal and transition clients.
///
/// # Errors
///
/// Returns an error if telemetry initialisation fails, command execution fails,
/// or output writing fails.
pub async fn run_with_clients<W>(
    command: Command,
    out: &mut W,
    terminal_client: &dyn TerminalClient,
    transition_service: Box<dyn TransitionService>,
    output_format: OutputFormat,
) -> Result<()>
where
    W: io::Write,
{
    run_with_clients_and_log_level(
        command,
        out,
        terminal_client,
        transition_service,
        None,
        output_format,
    )
    .await
}

/// Runs a command with injected clients and explicit telemetry settings.
///
/// # Errors
///
/// Returns an error if telemetry initialisation fails, command execution fails,
/// or output writing fails.
#[instrument(
    skip(out, terminal_client, transition_service),
    level = "info",
    fields(command = %command_name(&command), ?log_level, ?output_format)
)]
pub async fn run_with_clients_and_log_level<W>(
    command: Command,
    out: &mut W,
    terminal_client: &dyn TerminalClient,
    transition_service: Box<dyn TransitionService>,
    log_level: Option<LogLevel>,
    output_format: OutputFormat,
) -> Result<()>
where
    W: io::Write,
{
    telemetry::initialise_tracing(
        "shatter",
        terminal_client.stderr_is_terminal(),
        log_level.map(LogLevel::as_level_filter),
        output_format,
    )?;

    match command {
        Command::Render(args) => {
            crate::cli::render::run(transition_service.as_ref(), &args, out, output_format)
        }
    }
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::Render(_args) => "render",
    }
}
