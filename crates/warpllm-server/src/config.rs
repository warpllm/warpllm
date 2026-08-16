//! Server configuration: built-in defaults overridden by command-line flags.

use std::path::PathBuf;

use clap::Parser;
use warpllm::ClientConfig;

/// An OpenAI-compatible gateway: point any OpenAI SDK's base URL at it. The
/// gateway authenticates upstream with its own provider keys from the
/// environment, so the SDK's own key is ignored — pass any placeholder to
/// satisfy clients that insist on one.
#[derive(Debug, Parser)]
#[command(name = "warpllm", version)]
pub struct ServerConfig {
    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    /// Listen port
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// A roster of your own, in the same schema as warpllm's built-in
    /// `specs.yaml`, merged over it. How a self-hosted OpenAI-compatible
    /// server — vLLM, TGI, Ollama, llama.cpp — becomes routable. The built-in
    /// providers survive the merge; an entry naming one of them replaces it
    /// whole. Also read from `WARPLLM_SPECS` when this is not given.
    #[arg(long, value_name = "PATH")]
    pub specs: Option<PathBuf>,
    /// Upstream request timeout in seconds
    #[arg(long, default_value_t = 600)]
    pub timeout_secs: u64,
    /// Seconds a stream may go without a byte before it is abandoned
    /// (default: unbounded). Must exceed the slowest time-to-first-token you
    /// expect, since the wait before the first chunk is a gap like any other.
    #[arg(long)]
    pub stream_read_timeout_secs: Option<u64>,
}

/// Outcome of CLI parsing: run with a config, or print `text` and exit 0
/// (`--help` / `--version`).
pub enum Cli {
    Run(ServerConfig),
    Print(String),
}

/// The single flag parser shared by every CLI surface (binary, npx, PyPI).
/// Wrappers pass argv (program name stripped) straight through instead of
/// parsing flags themselves; the in-process ones can't let clap exit, so
/// help/version come back as [`Cli::Print`] and errors as `Err`.
pub fn parse_cli(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    match ServerConfig::try_parse_from(std::iter::once("warpllm".to_string()).chain(args)) {
        Ok(config) => Ok(Cli::Run(config)),
        // Not errors: clap models --help/--version as Err(DisplayHelp/...).
        Err(e) if !e.use_stderr() => Ok(Cli::Print(e.to_string())),
        Err(e) => Err(e.to_string()),
    }
}

impl ServerConfig {
    /// No key is set here: the client reads one per provider from the
    /// environment (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, …) when it is built,
    /// and `base_url` stays absent so every provider talks to its own API —
    /// which includes any address `--specs` named.
    ///
    /// `providers` stays absent too, so the gateway serves the whole roster it
    /// loaded, `--specs` entries included. Narrowing it is a client-side
    /// capability the gateway does not expose today — a key on the command line
    /// would sit in `ps` for a benefit the environment already provides.
    pub fn client_config(&self) -> ClientConfig {
        ClientConfig {
            base_url: None,
            specs_path: self.specs.clone(),
            timeout_secs: Some(self.timeout_secs),
            stream_read_timeout_secs: self.stream_read_timeout_secs,
            providers: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(list: &[&str]) -> Result<Cli, String> {
        parse_cli(list.iter().map(|s| s.to_string()))
    }

    fn config(list: &[&str]) -> ServerConfig {
        match parse(list) {
            Ok(Cli::Run(config)) => config,
            Ok(Cli::Print(_)) => panic!("expected Cli::Run, got Cli::Print"),
            Err(e) => panic!("expected Cli::Run, got Err: {e}"),
        }
    }

    #[test]
    fn no_flags_yield_defaults() {
        let config = config(&[]);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.timeout_secs, 600);
        // Unbounded by default: no value is right for every deployment, and
        // one set too tight cuts off a model that pauses to think.
        assert_eq!(config.stream_read_timeout_secs, None);
        assert_eq!(config.client_config().stream_read_timeout_secs, None);
    }

    /// The flag has to reach the CLIENT config, not just parse — it is the
    /// only way a gateway operator can bound a wedged upstream at all.
    #[test]
    fn the_stream_read_timeout_reaches_the_client_config() {
        let config = config(&["--stream-read-timeout-secs", "45"]);
        assert_eq!(config.stream_read_timeout_secs, Some(45));
        assert_eq!(config.client_config().stream_read_timeout_secs, Some(45));
    }

    /// The same, for the flag whose whole purpose is to reach the client: a
    /// `--specs` that parsed and went no further would leave the gateway
    /// silently routing against the built-in roster alone, and the only
    /// symptom would be a self-hosted model reported as unregistered.
    #[test]
    fn the_specs_flag_reaches_the_client_config() {
        let config = config(&["--specs", "./warpllm.yaml"]);
        assert_eq!(
            config.client_config().specs_path,
            Some(PathBuf::from("./warpllm.yaml"))
        );
    }

    /// Absent means the built-in roster, and the client falls back to
    /// `WARPLLM_SPECS` on its own — this must not manufacture a path.
    #[test]
    fn no_specs_flag_names_no_roster() {
        assert_eq!(config(&[]).client_config().specs_path, None);
    }

    #[test]
    fn flags_override_defaults() {
        let config = config(&[
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
            "--timeout-secs",
            "30",
        ]);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9090);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn equals_syntax_works() {
        assert_eq!(config(&["--port=9090"]).port, 9090);
    }

    #[test]
    fn unknown_flag_errors() {
        let err = parse(&["--verbose"]).map(|_| ()).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        assert!(err.contains("--verbose"), "{err}");
    }

    #[test]
    fn missing_value_errors() {
        let err = parse(&["--port"]).map(|_| ()).unwrap_err();
        assert!(err.contains("--port"), "{err}");
    }

    #[test]
    fn unparseable_value_errors() {
        let err = parse(&["--port", "eighty"]).map(|_| ()).unwrap_err();
        assert!(err.contains("invalid value 'eighty'"), "{err}");
    }

    #[test]
    fn help_lists_every_flag_with_defaults() {
        let Ok(Cli::Print(text)) = parse(&["--help"]) else {
            panic!("expected Cli::Print");
        };
        for expected in [
            "--host",
            "--port",
            "--specs",
            "--timeout-secs",
            "--stream-read-timeout-secs",
            "8080",
            "600",
        ] {
            assert!(text.contains(expected), "help missing {expected}: {text}");
        }
    }

    #[test]
    fn version_prints_the_workspace_version() {
        let Ok(Cli::Print(text)) = parse(&["--version"]) else {
            panic!("expected Cli::Print");
        };
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
    }
}
