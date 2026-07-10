use napi_derive::napi;

#[napi]
pub fn version() -> &'static str {
    warpllm::version()
}

#[napi]
pub async fn echo(msg: String) -> napi::Result<String> {
    warpllm::echo(&msg)
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}
