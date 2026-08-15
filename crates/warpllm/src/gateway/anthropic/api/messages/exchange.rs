//! The upstream half of a message exchange: gateway types in, gateway types
//! out.

use crate::error::Result;
use crate::gateway::anthropic::error::error_from_body;
use crate::gateway::types;
use crate::protocol::anthropic::messages::transport::{self, Outcome};

use super::{ingest_response, render_request};

/// Renders the gateway request for this protocol, posts it, and ingests the
/// reply — the one place this protocol's order is stated.
///
/// Gateway types on both ends, which is what lets every protocol implement this
/// same shape so `client.rs` gains one match arm per protocol and nothing else.
/// The argument LIST is not part of that invariant: `max_output_tokens` is here
/// because Anthropic requires a `max_tokens` and the gateway form's is optional,
/// so a ceiling has to reach the renderer from the roster when the caller named
/// none.
///
/// Stateless, taking the transport context loose rather than borrowing a
/// client: nothing here needs to outlive the call.
///
/// Error mapping happens here rather than in the transport because which
/// [`crate::Error`] a status becomes is a protocol-and-provider decision, while
/// reading the socket is not.
pub(crate) async fn exchange(
    request: &types::ChatRequest,
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    max_output_tokens: Option<u32>,
) -> Result<types::ChatResponse> {
    let wire = render_request(request, provider, max_output_tokens)?;
    match transport::post(http, provider, base_url, api_key, &wire).await? {
        Outcome::Ok(response) => Ok(ingest_response(response)),
        Outcome::Status {
            status,
            body,
            retry_after,
            request_id,
        } => Err(error_from_body(
            provider,
            status,
            &body,
            retry_after,
            request_id,
        )),
    }
}
