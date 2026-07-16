//! `POST /chat/completions` against OpenAI.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::http::{network_error, read_body};
use crate::providers::openai::{PROVIDER, error_from_body};
use crate::types::openai::chat::completions::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};

pub(crate) async fn post(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &CreateChatCompletionRequest,
) -> Result<CreateChatCompletionResponse> {
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

    let mut completion: CreateChatCompletionResponse =
        serde_json::from_str(&text).map_err(|e| Error::Decode {
            provider: PROVIDER,
            message: e.to_string(),
        })?;
    completion.model = request.model.clone();
    Ok(completion)
}
