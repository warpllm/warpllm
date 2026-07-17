use warpllm_server::config::{Cli, parse_cli};
use warpllm_server::{init_tracing, serve};

#[tokio::main]
async fn main() {
    let config = match parse_cli(std::env::args().skip(1)) {
        Ok(Cli::Print(text)) => {
            print!("{text}");
            return;
        }
        Ok(Cli::Run(config)) => config,
        Err(e) => {
            eprint!("{e}");
            std::process::exit(2);
        }
    };
    init_tracing();
    if let Err(e) = serve(config, shutdown_signal()).await {
        tracing::error!("server failed: {e}");
        std::process::exit(1);
    }
}

/// Resolves on SIGINT or, on unix, SIGTERM (what `docker stop` sends).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }
}
