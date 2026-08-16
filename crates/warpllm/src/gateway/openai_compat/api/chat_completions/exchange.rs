//! The upstream half of a chat completion: gateway types in, gateway types out.

use std::time::Duration;

use crate::auth::Authenticator;
use crate::error::Result;
use crate::gateway::openai_compat::error::error_from_body;
use crate::gateway::types;
use crate::protocol::openai_compat::chat_completions::transport::{
    self, ChunkStream, Outcome, StreamOutcome,
};

use super::{ingest_chunk, ingest_response, render_request};

/// Renders the gateway request for this protocol, posts it, and ingests the
/// reply — the one place this protocol's order is stated.
///
/// Gateway types on both ends, deliberately: that is what lets every protocol
/// implement this same signature, so `client.rs` gains one match arm per
/// protocol and nothing else. The caller-side conversions stay with the client,
/// since they answer to the protocol warpllm was *called* with, not this one.
///
/// Stateless, taking the transport context loose rather than borrowing a client:
/// nothing here needs to outlive the call.
///
/// Error mapping happens here rather than in the transport because which
/// [`crate::Error`] a status becomes is a protocol-and-provider decision, while
/// reading the socket is not.
pub(crate) async fn exchange(
    request: &types::ChatRequest,
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: Option<&Authenticator>,
) -> Result<types::ChatResponse> {
    let wire = render_request(request, provider)?;
    match transport::post(http, provider, base_url, auth, &wire).await? {
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

/// [`exchange`] for a streamed reply: gateway types on both ends, the same
/// order stated once, and the same error mapping.
///
/// The failure it can report is only the one that happens BEFORE any chunk —
/// a refused request, a rate limit, an unreachable host. Once the stream is
/// open, a failure has no status left to map and surfaces on the stream itself.
///
/// `read_timeout` travels through rather than being read from a client here,
/// which is what keeps this stateless.
pub(crate) async fn exchange_stream(
    request: &types::ChatRequest,
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: Option<&Authenticator>,
    read_timeout: Option<Duration>,
) -> Result<ChatChunkStream> {
    let wire = render_request(request, provider)?;
    match transport::post_stream(http, provider, base_url, auth, &wire, read_timeout).await? {
        StreamOutcome::Ok(chunks) => Ok(ChatChunkStream { chunks }),
        StreamOutcome::Status {
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

/// The gateway's view of a stream: wire chunks in, [`types::ChatResponseChunk`]
/// out, one for one.
///
/// Thin on purpose. Ingest is infallible, so this adds nothing to the transport
/// but the conversion — which is exactly the seam a second protocol needs, and
/// the reason this is not just [`ChunkStream`] handed to the client.
#[derive(Debug)]
pub(crate) struct ChatChunkStream {
    chunks: ChunkStream,
}

impl ChatChunkStream {
    pub(crate) async fn next(&mut self) -> Option<Result<types::ChatResponseChunk>> {
        Some(self.chunks.next().await?.map(ingest_chunk))
    }
}
