//! Shared HTTP helpers used by every provider implementation.

use crate::error::{Error, Result};

/// Maps a transport-level reqwest error to [`Error::Network`].
pub(crate) fn network_error(provider: &'static str, source: reqwest::Error) -> Error {
    Error::Network { provider, source }
}

/// Reads the response body, mapping read failures to [`Error::Network`].
pub(crate) async fn read_body(
    provider: &'static str,
    response: reqwest::Response,
) -> Result<(u16, String)> {
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|e| network_error(provider, e))?;
    Ok((status, body))
}
