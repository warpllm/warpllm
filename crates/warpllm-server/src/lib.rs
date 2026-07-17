//! OpenAI-compatible HTTP gateway over the warpllm client.
//!
//! Point any OpenAI SDK's `base_url` at this server. The gateway holds the
//! provider keys (`OPENAI_API_KEY` from the environment for now; a
//! configuration surface later) — the caller's own Authorization header is
//! ignored, like bifrost and litellm.

pub mod config;
pub mod error;
mod handlers;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::Level;
use tracing_subscriber::EnvFilter;

use crate::config::ServerConfig;

#[derive(Clone)]
pub struct AppState {
    pub client: Arc<warpllm::Client>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        // Image-bearing messages exceed axum's 2 MB default.
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        // Outermost layer (added last): a span per request with method,
        // path, status, and latency. Explicit INFO — the tower-http
        // defaults are DEBUG, invisible under the default filter.
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state)
}

/// Installs the fmt tracing subscriber honoring `RUST_LOG` (default `info`).
/// Idempotent: later calls (e.g. `serve` invoked twice from a binding) are
/// no-ops rather than panics.
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        // Color only on a real terminal; piped/collected logs (docker logs,
        // aggregators) must not get ANSI escapes.
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .try_init()
        .ok();
}

/// Binds `config.host:config.port` and serves the gateway until `shutdown`
/// resolves. Both the standalone binary and the language bindings enter
/// here; only signal handling differs between them.
pub async fn serve(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = warpllm::Client::new(config.client_config())?;
    let app = router(AppState {
        client: Arc::new(client),
    });
    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port)).await?;
    tracing::info!(
        addr = %listener.local_addr()?,
        version = warpllm::version(),
        "warpllm gateway listening"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
