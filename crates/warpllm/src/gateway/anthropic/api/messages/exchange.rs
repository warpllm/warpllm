//! The upstream half of a message exchange: gateway types in, gateway types
//! out.

use std::time::Duration;

use crate::auth::Authenticator;
use crate::error::Result;
use crate::gateway::anthropic::error::error_from_body;
use crate::gateway::types;
use crate::protocol::anthropic::messages::transport::{self, EventStream, Outcome, StreamOutcome};
use crate::protocol::anthropic::messages::types::MessageStreamEvent;

use super::stream::{StreamState, ingest_event};
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
    auth: &Authenticator,
    max_output_tokens: Option<u32>,
) -> Result<types::ChatResponse> {
    let wire = render_request(request, provider, max_output_tokens)?;
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
/// The failure it can report is only the one that happens BEFORE any event — a
/// refused request, a rate limit, an unreachable host. Once the stream is open
/// there is no status left to map, and a failure surfaces on the stream itself.
///
/// `read_timeout` travels through rather than being read from a client, which
/// is what keeps this stateless.
pub(crate) async fn exchange_stream(
    request: &types::ChatRequest,
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: &Authenticator,
    max_output_tokens: Option<u32>,
    read_timeout: Option<Duration>,
) -> Result<ChatChunkStream> {
    // `render_request` puts `stream: true` on the body from the gateway form;
    // the transport CHECKS it rather than setting it, so a request whose
    // gateway form says otherwise fails here rather than being silently
    // corrected. See `transport::send`.
    let wire = render_request(request, provider, max_output_tokens)?;
    match transport::post_stream(http, provider, base_url, auth, &wire, read_timeout).await? {
        StreamOutcome::Ok(events) => Ok(ChatChunkStream {
            events,
            provider,
            // The one thing the stream needs from the request, and the reason
            // it is read here: Anthropic reports a stream's totals whether or
            // not anyone asked, and only the request knows whether anyone did.
            state: StreamState::new(request.stream_include_usage == Some(true)),
        }),
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

/// The gateway's view of a stream: wire events in, [`types::ChatResponseChunk`]
/// out — but NOT one for one.
///
/// This is the seam its openai_compat counterpart says it exists for, and the
/// reason that one can be "thin on purpose" while this one cannot. Two things
/// differ, and both are this protocol's shape rather than an implementation
/// choice:
///
/// * **State.** `id` and `model` arrive once, on `message_start`, and every
///   gateway chunk requires them. [`StreamState`] is the memory that makes that
///   possible, and it lives here because a stream is the only scope that
///   outlives a single event.
/// * **Events that are not chunks.** `content_block_stop` and `ping` carry
///   nothing a chunk can hold, so they ingest to `None` and this loop skips
///   them rather than emitting empty values. `message_stop` joins them
///   whenever the caller did not ask for the stream's totals, which are the
///   only thing it has to say.
/// * **A request field the stream reads.** `stream_include_usage`, for the
///   same reason — see [`StreamState`].
/// * **An event that is a FAILURE.** See [`Self::next`].
#[derive(Debug)]
pub(crate) struct ChatChunkStream {
    events: EventStream,
    provider: &'static str,
    state: StreamState,
}

/// The status a mid-stream failure is classified against.
///
/// There is no real one: the HTTP exchange already answered 200 and the failure
/// arrived after it, which is exactly the case `error.rs`'s `type` table exists
/// for. Passing the status that WAS sent keeps `classify` honest — it finds no
/// status rule, falls to the family, and reports `overloaded_error` as an
/// overload rather than as an unattributed server fault.
const AFTER_THE_HEADERS: u16 = 200;

impl ChatChunkStream {
    /// The next chunk, skipping the events that carry no gateway content and
    /// converting the one that carries a failure.
    ///
    /// A loop rather than a single read: skipping is not the same as ending,
    /// and returning `None` for a `ping` would cut a live stream short at the
    /// first keepalive.
    ///
    /// The `error` arm is the reason this cannot be a plain skip. Anthropic
    /// sends errors mid-stream after a 200 — an overload during generation —
    /// and `EventStream` hands one over as `Ok(event)` with the stream marked
    /// COMPLETE, because to the transport it is a well-formed event that ends
    /// the stream. Skipping it the way a `ping` is skipped would read the next
    /// event, get `None` from a completed stream, and report a truncated reply
    /// as a finished one — burying the reason the provider gave under a clean
    /// EOF. So it becomes an [`Err`] here, at the one layer that knows a
    /// failure from an absence.
    pub(crate) async fn next(&mut self) -> Option<Result<types::ChatResponseChunk>> {
        loop {
            let event = match self.events.next().await? {
                Ok(event) => event,
                Err(error) => return Some(Err(error)),
            };
            if let MessageStreamEvent::Error(_) = &event {
                return Some(Err(self.failure(&event)));
            }
            if let Some(chunk) = ingest_event(event, &mut self.state) {
                return Some(Ok(chunk));
            }
        }
    }

    /// A mid-stream `error` event as the failure it reports.
    ///
    /// Serialized whole rather than reaching into the payload: `error_from_body`
    /// reads the `{"type": "error", "error": {…}}` envelope this is, so the
    /// mid-stream and the non-2xx paths classify one shape through one table.
    fn failure(&self, event: &MessageStreamEvent) -> crate::Error {
        let body = serde_json::to_string(event).unwrap_or_default();
        error_from_body(self.provider, AFTER_THE_HEADERS, &body, None, None)
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::auth::Authenticator;
    use crate::gateway::types::{GenerationParams, Role};

    const START: &str = concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","#,
        r#""role":"assistant","model":"claude-opus-5","content":[],"#,
        r#""stop_reason":null,"stop_sequence":null,"#,
        r#""usage":{"input_tokens":4,"output_tokens":1}}}"#,
        "\n\n",
    );

    fn request() -> types::ChatRequest {
        types::ChatRequest {
            model: "claude-opus-5".into(),
            messages: vec![types::Message {
                role: Role::User,
                content: vec![types::ContentBlock::Text {
                    text: "hi".into(),
                    cache: None,
                }],
                ext: types::ProviderExt::new(),
            }],
            // The transport CHECKS this rather than setting it, so the gateway
            // form has to say so before `render_request` can put it on the body.
            stream: true,
            params: GenerationParams {
                max_tokens: Some(16),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Opens a stream against a body the mock serves verbatim.
    async fn stream_of(body: String) -> ChatChunkStream {
        opened(request(), body).await
    }

    /// [`stream_of`] for a request the caller shaped, which is the only way to
    /// reach what the stream reads OFF that request.
    async fn opened(request: types::ChatRequest, body: String) -> ChatChunkStream {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let stream = exchange_stream(
            &request,
            &reqwest::Client::new(),
            "anthropic",
            &server.uri(),
            &Authenticator::anthropic_api_key("sk-ant-demo".into()),
            None,
            None,
        )
        .await
        .expect("a 200 opens a stream");
        // The mock server must outlive the socket.
        std::mem::forget(server);
        stream
    }

    async fn collect(mut stream: ChatChunkStream) -> Vec<Result<types::ChatResponseChunk>> {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }
        chunks
    }

    /// THE failure this seam exists to prevent. Anthropic reports an overload
    /// mid-generation as an `error` EVENT after a 200, and the transport hands
    /// it over as a well-formed event that ends the stream. A reader that
    /// skipped it the way it skips a `ping` would then read a completed stream,
    /// get `None`, and report a half-written reply as a finished one — the
    /// provider's reason buried under a clean EOF.
    #[tokio::test]
    async fn a_mid_stream_error_is_a_failure_and_not_a_clean_end() {
        let chunks = collect(
            stream_of(format!(
                "{START}event: error\ndata: {}\n\n",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
            ))
            .await,
        )
        .await;

        let last = chunks.last().expect("the stream produced nothing at all");
        let error = last
            .as_ref()
            .expect_err("the mid-stream error was reported as a successful chunk");
        assert!(
            matches!(error, crate::Error::Overloaded(_)),
            "the family decided the meaning: {error}"
        );
        assert!(error.to_string().contains("Overloaded"), "{error}");
    }

    /// The events that carry nothing ARE skipped, which is the behaviour the
    /// error arm above must not be confused with. A `ping` between two content
    /// events must not end the stream or produce an empty chunk.
    #[tokio::test]
    async fn keepalives_and_block_boundaries_are_skipped_not_ended() {
        let chunks = collect(
            stream_of(format!(
                "{START}\
                 event: ping\ndata: {{\"type\":\"ping\"}}\n\n\
                 event: content_block_start\ndata: {}\n\n\
                 event: ping\ndata: {{\"type\":\"ping\"}}\n\n\
                 event: content_block_delta\ndata: {}\n\n\
                 event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                 event: message_delta\ndata: {}\n\n\
                 event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
            ))
            .await,
        )
        .await;

        assert!(
            chunks.iter().all(Result::is_ok),
            "a well-formed stream reported a failure"
        );
        // message_start, block start, block delta, message delta — and NOT the
        // two pings, the block stop, or the message stop, which carries the
        // totals this request did not ask for. See the test below.
        assert_eq!(chunks.len(), 4, "contentless events produced chunks");
        assert_eq!(
            chunks.last().unwrap().as_ref().unwrap().completions[0]
                .finish_reason_raw
                .as_deref(),
            Some("end_turn")
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.as_ref().unwrap().usage.is_none()),
            "totals nobody asked for reached the caller"
        );
    }

    /// And the same transcript, for a caller who DID ask, grows one chunk: the
    /// totals, on a chunk of their own.
    ///
    /// The gating lives in [`StreamState`] and the decision to apply it lives
    /// HERE, in the one place that can see the request — so a test against the
    /// state alone would pass with this seam reading nothing at all.
    #[tokio::test]
    async fn a_caller_who_asked_for_the_totals_gets_them_on_a_chunk_of_their_own() {
        let chunks = collect(
            opened(
                types::ChatRequest {
                    stream_include_usage: Some(true),
                    ..request()
                },
                format!(
                    "{START}\
                     event: message_delta\ndata: {}\n\n\
                     event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
                ),
            )
            .await,
        )
        .await;

        // message_start, message_delta, and the totals `message_stop` carries.
        assert_eq!(chunks.len(), 3);
        let totals = chunks.last().unwrap().as_ref().unwrap();
        assert!(totals.completions.is_empty());
        let usage = totals.usage.as_ref().expect("the totals were asked for");
        assert_eq!(usage.output_tokens, Some(9));
        assert_eq!(usage.input_tokens, Some(4));
    }
}
