use thiserror::Error;

/// Errors returned by telemetry initialisation.
#[derive(Debug, Error)]
pub(crate) enum TelemetryError {
    #[error("failed to install tracing subscriber")]
    Subscriber(#[from] tracing_subscriber::util::TryInitError),
}
