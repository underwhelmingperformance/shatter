use std::io::IsTerminal;
use std::process::ExitCode;

use clap::Parser;

use shatter::{Args, OutputFormat, RealTransitionService, run_with_log_level};

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    let run_result = async {
        let log_level = args.log_level();
        let output_format = args.output_format().unwrap_or(if stdout.is_terminal() {
            OutputFormat::Pretty
        } else {
            OutputFormat::Json
        });
        let command = args.into_command()?;

        run_with_log_level(
            command,
            &mut stdout,
            Box::new(RealTransitionService::new()),
            log_level,
            output_format,
        )
        .await
    }
    .await;

    match run_result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if let Some(clap_error) = error.downcast_ref::<clap::Error>() {
                let _ = clap_error.print();
                return if clap_error.use_stderr() {
                    ExitCode::from(2)
                } else {
                    ExitCode::SUCCESS
                };
            }
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}
