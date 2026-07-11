//! Client configuration with env-var fallback for API keys.

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    pub openai_api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
}

impl ClientConfig {
    /// Fills a missing key from `OPENAI_API_KEY`. Explicit values win.
    /// Delegates to [`merge_env`] so the precedence logic stays a pure,
    /// testable function.
    pub(crate) fn resolve_env(self) -> Self {
        merge_env(self, std::env::var("OPENAI_API_KEY").ok())
    }
}

pub(crate) fn merge_env(mut config: ClientConfig, openai_env: Option<String>) -> ClientConfig {
    config.openai_api_key = config.openai_api_key.or(openai_env);
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_fills_missing_key() {
        let config = merge_env(ClientConfig::default(), Some("sk-env".into()));
        assert_eq!(config.openai_api_key.as_deref(), Some("sk-env"));
    }

    #[test]
    fn explicit_key_wins_over_env() {
        let config = merge_env(
            ClientConfig {
                openai_api_key: Some("sk-explicit".into()),
                ..Default::default()
            },
            Some("sk-env".into()),
        );
        assert_eq!(config.openai_api_key.as_deref(), Some("sk-explicit"));
    }

    #[test]
    fn missing_env_keys_do_not_panic() {
        let config = merge_env(ClientConfig::default(), None);
        assert_eq!(config.openai_api_key, None);
    }
}
