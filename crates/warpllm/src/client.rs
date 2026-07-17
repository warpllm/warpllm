//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::model::{Provider, parse_model};
use crate::providers;
use crate::types::openai::chat::completions::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    /// Resolved once, at construction, from `OPENAI_API_KEY`.
    api_key: Option<String>,
}

impl Client {
    /// The API key comes from `OPENAI_API_KEY` (read once, at construction),
    /// exactly like the OpenAI SDKs. A missing key only errors at request
    /// time, so constructing a client never requires credentials.
    pub fn new(config: ClientConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(Self {
            http,
            config,
            api_key: std::env::var("OPENAI_API_KEY").ok(),
        })
    }

    /// A copy of this client that authenticates with `api_key` instead of
    /// the environment's. Cheap — the connection pool is shared — so
    /// gateways call it per request to forward each caller's bearer token
    /// upstream.
    #[must_use]
    pub fn with_api_key(&self, api_key: impl Into<String>) -> Self {
        Self {
            http: self.http.clone(),
            config: self.config.clone(),
            api_key: Some(api_key.into()),
        }
    }

    pub async fn chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        if request.stream == Some(true) {
            return Err(Error::NotImplemented("streaming"));
        }
        let (provider, model) = parse_model(&request.model)?;
        match provider {
            Provider::OpenAi => {
                let api_key = self.openai_key()?;
                providers::openai::chat::completions::post(
                    &self.http,
                    self.openai_base_url(),
                    api_key,
                    model,
                    &request,
                )
                .await
            }
        }
    }

    fn openai_key(&self) -> Result<&str> {
        self.api_key.as_deref().ok_or(Error::MissingApiKey {
            provider: "openai",
            env_var: "OPENAI_API_KEY",
        })
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    fn openai_base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(providers::openai::DEFAULT_BASE_URL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_defaults_to_the_provider_api() {
        let client = Client::new(ClientConfig::default()).unwrap();
        assert_eq!(client.openai_base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn configured_base_url_wins_over_the_default() {
        let client = Client::new(ClientConfig {
            base_url: Some("http://localhost:9999".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(client.openai_base_url(), "http://localhost:9999");
    }
}
