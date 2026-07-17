//! Client configuration.

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Deliberately holds no API key: the client reads `OPENAI_API_KEY` from the
/// environment (exactly like the OpenAI SDKs), and gateways forward each
/// caller's bearer via [`crate::Client::with_api_key`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// Overrides the provider's default base URL (proxies, tests). Absent
    /// means each provider talks to its own API.
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
}
