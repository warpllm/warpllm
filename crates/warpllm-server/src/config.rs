//! Server configuration: built-in defaults overridden by command-line flags.

use clap::Parser;
use warpllm::ClientConfig;

/// An OpenAI-compatible gateway: point any OpenAI SDK's base URL at it and
/// pass your provider API key as the bearer token.
#[derive(Debug, Parser)]
#[command(name = "warpllm", version)]
pub struct ServerConfig {
    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    /// Listen port
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Upstream request timeout in seconds
    #[arg(long, default_value_t = 600)]
    pub timeout_secs: u64,
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
    /// Auth is passthrough (each caller's bearer becomes the upstream key,
    /// with the gateway's own `OPENAI_API_KEY` env as the client fallback)
    /// and `base_url` stays absent so every provider talks to its own API.
    pub fn client_config(&self) -> ClientConfig {
        ClientConfig {
            base_url: None,
            timeout_secs: Some(self.timeout_secs),
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
        for expected in ["--host", "--port", "--timeout-secs", "8080", "600"] {
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
