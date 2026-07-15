//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::model::{ProviderKind, parse_model};
use crate::providers;
use crate::types::openai::chat::completions::{ChatCompletion, ChatCompletionRequest};

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
}

impl Client {
    /// Missing API keys are filled from `OPENAI_API_KEY` (explicit values
    /// win). A key missing for a provider only errors at request time, so
    /// constructing a client never requires credentials.
    pub fn new(config: ClientConfig) -> Result<Self> {
        let config = config.resolve_env();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(Self { http, config })
    }

    pub async fn chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletion> {
        if request.stream == Some(true) {
            return Err(Error::NotImplemented("streaming coming in a later release"));
        }
        let (provider, model) = parse_model(&request.model)?;
        match provider {
            ProviderKind::OpenAi => {
                let api_key =
                    self.config
                        .openai_api_key
                        .as_deref()
                        .ok_or(Error::MissingApiKey {
                            provider: "openai",
                            env_var: "OPENAI_API_KEY",
                        })?;
                let base = self
                    .config
                    .base_url
                    .as_deref()
                    .ok_or_else(|| Error::InvalidInput("missing base_url".into()))?;
                providers::openai::chat::completions::post(
                    &self.http, base, api_key, model, &request,
                )
                .await
            }
        }
    }
}
