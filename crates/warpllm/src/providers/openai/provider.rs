//! OpenAI chat completions: the wire format already matches our types, so
//! this provider only strips the `openai/` prefix and maps errors.

use serde_json::Value;

use super::types::{ChatCompletion, ChatCompletionRequest};
use crate::error::{Error, Result};
use crate::http::{network_error, read_body};

const PROVIDER: &str = "openai";

pub(crate) async fn chat_completion(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletion> {
    let mut body = serde_json::to_value(request).map_err(|e| Error::InvalidInput(e.to_string()))?;
    body["model"] = Value::String(model.to_string());

    let response = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| network_error(PROVIDER, e))?;

    let (status, text) = read_body(PROVIDER, response).await?;
    if !(200..300).contains(&status) {
        return Err(error_from_body(status, &text));
    }

    let mut completion: ChatCompletion =
        serde_json::from_str(&text).map_err(|e| Error::Decode {
            provider: PROVIDER,
            message: e.to_string(),
        })?;
    completion.model = request.model.clone();
    Ok(completion)
}

/// OpenAI error bodies look like `{"error": {"message": ..., "type": ...}}`.
/// Unparseable bodies fall back to the (truncated) raw text.
fn error_from_body(status: u16, body: &str) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|v| &v["error"]);
    Error::Provider {
        provider: PROVIDER,
        status,
        error_type: error.and_then(|e| e["type"].as_str()).map(str::to_string),
        message: error
            .and_then(|e| e["message"].as_str())
            .map(str::to_string)
            .unwrap_or_else(|| truncate(body)),
    }
}

fn truncate(body: &str) -> String {
    const MAX: usize = 500;
    if body.len() <= MAX {
        body.to_string()
    } else {
        let mut end = MAX;
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &body[..end])
    }
}
