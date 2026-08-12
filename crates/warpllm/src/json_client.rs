//! The shared JSON boundary used by foreign-language bindings.
//!
//! PyO3 and napi-rs should expose Rust, not each maintain the same serde
//! adapter. Keeping that adapter here makes both native modules mechanical.

use crate::{Client, ClientConfig, CreateChatCompletionRequest, Error, Result};

/// A [`Client`] whose inputs and outputs are JSON strings.
///
/// This is intentionally small: ownership and async-runtime integration stay
/// language-specific, while parsing, validation, dispatch, and serialization
/// happen once in the core.
pub struct JsonClient {
    inner: Client,
}

impl JsonClient {
    pub fn new(config_json: &str) -> Result<Self> {
        let config: ClientConfig = serde_json::from_str(config_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        Ok(Self {
            inner: Client::new(config)?,
        })
    }

    pub async fn chat_completions(&self, request_json: &str) -> Result<String> {
        let request: CreateChatCompletionRequest = serde_json::from_str(request_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        // Refused HERE rather than inherited from [`Client::chat_completions`],
        // which refuses `stream: true` by naming the Rust method that serves
        // it. That advice is true of Rust and of nothing on this boundary: a
        // binding built on it has one method, and telling its caller to reach
        // for a second one they cannot see is worse than saying no.
        //
        // So what a binding caller is told is that the SURFACE does not stream,
        // which is the truth until this boundary can carry chunks. The HTTP
        // gateway states the same thing for the same reason.
        if request.stream == Some(true) {
            return Err(Error::NotImplemented(
                "streaming over the language bindings",
            ));
        }
        let response = self.inner.chat_completions(request).await?;
        serde_json::to_string(&response).map_err(|error| Error::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_config_is_a_core_error() {
        let error = JsonClient::new(r#"{"unknown": true}"#)
            .err()
            .expect("invalid configuration should fail");
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    /// A binding caller asking to stream is told the SURFACE cannot, not to
    /// call a Rust method it has no way to reach. Both language suites assert
    /// this from the other side; pinning it here is what keeps the Rust client
    /// free to give Rust callers better advice without silently changing what
    /// a Node or Python caller is told.
    #[tokio::test]
    async fn streaming_over_the_json_boundary_is_not_implemented() {
        let client = JsonClient::new("{}").unwrap();
        let error = client
            .chat_completions(r#"{"model":"openai/gpt-5.6","messages":[],"stream":true}"#)
            .await
            .expect_err("the boundary cannot carry chunks");
        assert!(matches!(error, Error::NotImplemented(_)), "{error:?}");
        // 501, and the word a caller greps for.
        assert_eq!(error.to_openai().status, Some(501));
        assert!(error.to_string().contains("streaming"), "{error}");
    }
}
