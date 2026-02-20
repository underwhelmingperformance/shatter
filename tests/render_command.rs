use std::sync::{Arc, Mutex};

use clap::Parser;
use pretty_assertions::assert_eq;

use shatter::{
    Args, Command, OutputFormat, PanelDimensions, RenderReceipt, RenderSize, TerminalClient,
    TransitionError, TransitionRequest, TransitionService, run_with_clients,
};

#[derive(Debug, Default)]
struct FakeTerminalClient;

impl TerminalClient for FakeTerminalClient {
    fn stdout_is_terminal(&self) -> bool {
        false
    }

    fn stderr_is_terminal(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
struct FakeTransitionService {
    captured: Arc<Mutex<Vec<TransitionRequest>>>,
}

impl FakeTransitionService {
    fn new() -> Self {
        Self::default()
    }

    fn captured(&self) -> Arc<Mutex<Vec<TransitionRequest>>> {
        Arc::clone(&self.captured)
    }
}

impl TransitionService for FakeTransitionService {
    fn render(&self, request: TransitionRequest) -> Result<RenderReceipt, TransitionError> {
        self.captured
            .lock()
            .expect("capture lock should be available")
            .push(request.clone());
        let dimensions = match request.size() {
            RenderSize::Auto => PanelDimensions::new(1, 1).expect("1x1 should be valid"),
            RenderSize::Fixed(dimensions) => dimensions,
        };

        Ok(RenderReceipt::new(
            request.output_path().to_path_buf(),
            dimensions,
            request.frame_count().get(),
            request.fps().get(),
        ))
    }
}

#[tokio::test]
async fn run_with_clients_emits_json_for_render_command() -> anyhow::Result<()> {
    let args = Args::try_parse_from([
        "shatter",
        "--output-format",
        "json",
        "render",
        "from.png",
        "to.png",
        "out.gif",
        "--size",
        "48x32",
        "--frames",
        "12",
        "--fps",
        "10",
        "--seed",
        "99",
    ])?;
    let command = args.into_command()?;

    let service = FakeTransitionService::new();
    let captured = service.captured();
    let mut out = Vec::new();

    run_with_clients(
        command,
        &mut out,
        &FakeTerminalClient,
        Box::new(service),
        OutputFormat::Json,
    )
    .await?;

    let rendered = String::from_utf8(out)?;
    let expected = r#"{
  "action": "rendered",
  "output": "out.gif",
  "dimensions": "48x32",
  "frames": 12,
  "fps": 10
}
"#;
    assert_eq!(expected, rendered);

    let captured_guard = captured
        .lock()
        .expect("capture lock should be available after run");
    let captured_requests = captured_guard.as_slice();
    assert_eq!(1, captured_requests.len());
    assert_eq!(
        RenderSize::Fixed(PanelDimensions::new(48, 32).expect("48x32 should be valid")),
        captured_requests[0].size()
    );
    assert_eq!(12, captured_requests[0].frame_count().get());
    assert_eq!(10, captured_requests[0].fps().get());
    assert_eq!(99, captured_requests[0].seed());

    Ok(())
}

#[test]
fn args_roundtrip_to_render_command() -> anyhow::Result<()> {
    let args = Args::try_parse_from(["shatter", "render", "first.png", "second.png", "out.gif"])?;

    let command = args.into_command()?;
    assert!(matches!(command, Command::Render(_)));
    Ok(())
}

#[tokio::test]
async fn run_with_clients_defaults_to_auto_size_mode() -> anyhow::Result<()> {
    let args = Args::try_parse_from(["shatter", "render", "from.png", "to.png", "out.gif"])?;
    let command = args.into_command()?;

    let service = FakeTransitionService::new();
    let captured = service.captured();
    let mut out = Vec::new();

    run_with_clients(
        command,
        &mut out,
        &FakeTerminalClient,
        Box::new(service),
        OutputFormat::Json,
    )
    .await?;

    let captured_guard = captured
        .lock()
        .expect("capture lock should be available after run");
    let captured_requests = captured_guard.as_slice();
    assert_eq!(1, captured_requests.len());
    assert_eq!(RenderSize::Auto, captured_requests[0].size());

    Ok(())
}
