//! The HTTP binding for `POST /messages`: the one place the
//! [`Api::AnthropicMessages`](crate::Api) → URL mapping physically lives, since
//! the module path spells the API's name rather than its route.
//!
//! Everything Anthropic does differently at the HTTP layer is in this file and
//! nowhere else — the path, the key header, and the version header:
//!
//! | | openai_compat | anthropic |
//! |---|---|---|
//! | path | `{base_url}/chat/completions` | `{base_url}/messages` |
//! | auth | `Authorization: Bearer …` | `x-api-key: …` |
//! | extra | — | `anthropic-version: 2023-06-01` |
//!
//! That containment is the point: Bedrock (#24) and Vertex (#25) speak the same
//! Messages shapes over different transports and auth, so they contribute a
//! sibling of this file and reuse [`types`](super::types) whole.

use std::time::Duration;

use crate::error::{Error, Result};
use crate::http::{SseFrames, network_error, read_response};
use crate::protocol::anthropic::messages::types::{
    CreateMessageRequest, Message, MessageStreamEvent,
};

/// The API version this client is written against, sent on every request.
///
/// Anthropic pins breaking changes behind this header rather than a URL
/// segment, so it is not optional: a request without it is rejected, and a
/// request with a different one may be answered in shapes
/// [`types`](super::types) does not model.
const API_VERSION: &str = "2023-06-01";

/// What the upstream said. A non-2xx is NOT an [`Err`] here: which [`Error`] a
/// given status and body becomes is the caller's to decide, and deciding here
/// would mean `protocol` reaching into the conversion layer to find out, which
/// is the dependency this module exists without. `Err` is reserved for failures
/// nothing could reinterpret — the request never completing, or a 2xx body that
/// will not decode.
///
/// `large_enum_variant` is allowed rather than fixed for the same reason its
/// counterpart is: boxing the success variant would add an allocation to every
/// successful request to shrink a value constructed once, moved once, and
/// destructured immediately.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Outcome {
    Ok(Message),
    /// The status, raw body, and header evidence, verbatim, for the caller to
    /// map.
    ///
    /// The headers travel with the body because this is the LAST place they
    /// exist: `retry-after` and the upstream request id appear nowhere a
    /// caller handed only a status and a body could reach.
    Status {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
        request_id: Option<String>,
    },
}

/// Sends a request and reads the whole reply.
///
/// REQUIRES a body that does not ask for a stream, and says so rather than
/// quietly making it true — see [`post_stream`] for why the check is here at
/// all, and why it refuses rather than corrects.
pub(crate) async fn post(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateMessageRequest,
) -> Result<Outcome> {
    if body.stream == Some(true) {
        return Err(Error::InvalidInput(
            "stream: true asks for events; a whole reply cannot carry them".into(),
        ));
    }
    let response = request(http, base_url, api_key, body)
        .send()
        .await
        .map_err(|e| network_error(provider, e))?;

    let parts = read_response(provider, response).await?;
    if !(200..300).contains(&parts.status) {
        return Ok(Outcome::Status {
            status: parts.status,
            body: parts.body,
            retry_after: parts.retry_after,
            request_id: parts.request_id,
        });
    }

    serde_json::from_str(&parts.body)
        .map(Outcome::Ok)
        .map_err(|e| Error::Decode {
            provider,
            message: e.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Streaming.
//
// Everything below here is still unreachable: the conversions that call `post`
// have landed, and the ones that read events have not. Each item carries its
// own `allow` rather than the module carrying one, so that a NON-streaming item
// going dead is still a warning — and so that deleting these is exactly the
// change that wires the stream up.
// ---------------------------------------------------------------------------

/// [`Outcome`] for a streamed request. Same contract: a non-2xx is DATA, and
/// only the caller decides which [`Error`] it becomes.
///
/// The success arm holds an open socket rather than a decoded body, which is
/// the whole difference between this and [`Outcome`]: nothing has been read
/// past the headers, so the status decision is made before the first event and
/// never again.
#[derive(Debug)]
#[allow(dead_code)] // staged: read once the stream conversions land
pub(crate) enum StreamOutcome {
    Ok(EventStream),
    /// The status, raw body, and header evidence, verbatim — see
    /// [`Outcome::Status`].
    Status {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
        request_id: Option<String>,
    },
}

/// Sends a `stream: true` request and hands back the events, undecoded until
/// asked for.
///
/// REQUIRES a body that already asks for a stream, and REFUSES one that does
/// not rather than setting the flag on its behalf — see [`request`].
///
/// A non-2xx body is read to the end here, exactly as [`post`] does: an error
/// reply is small, complete, and worth nothing streamed.
///
/// `read_timeout` bounds the GAP between reads once the stream is open. It is
/// applied here rather than on the [`reqwest::Client`] because reqwest's own
/// `read_timeout` is builder-scoped: setting it there would bind every
/// non-streamed request too, or cost a second connection pool to avoid that.
#[allow(dead_code)] // staged: read once the stream conversions land
pub(crate) async fn post_stream(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateMessageRequest,
    read_timeout: Option<Duration>,
) -> Result<StreamOutcome> {
    if body.stream != Some(true) {
        return Err(Error::InvalidInput(
            "a streamed request must carry stream: true".into(),
        ));
    }
    let response = request(http, base_url, api_key, body)
        .send()
        .await
        .map_err(|e| network_error(provider, e))?;

    if !(200..300).contains(&response.status().as_u16()) {
        let parts = read_response(provider, response).await?;
        return Ok(StreamOutcome::Status {
            status: parts.status,
            body: parts.body,
            retry_after: parts.retry_after,
            request_id: parts.request_id,
        });
    }

    Ok(StreamOutcome::Ok(EventStream {
        response,
        provider,
        // No sentinel: Anthropic ends a stream with a `message_stop` EVENT,
        // which only a decode can recognize. See `ends_the_stream`.
        frames: SseFrames::new(None),
        ended: false,
        complete: false,
        truncated: false,
        read_timeout,
    }))
}

/// The request both entry points send, differing only in the body's `stream`.
///
/// Which each of them CHECKS rather than sets. Getting it wrong fails silently
/// in the direction that matters: a `post_stream` whose body says
/// `stream: false` gets an ordinary JSON `Message` back, which [`EventStream`]
/// then reads as SSE — no `data:` line ever dispatches, the socket closes, and
/// the caller is told the reply was TRUNCATED. A complete, correct answer
/// reported as a broken connection, with nothing in the logs pointing at the
/// flag. (The mirror case is merely loud: SSE handed to [`post`] fails to
/// decode and says so.)
///
/// Checking rather than setting because a body that disagrees with the
/// function called for it is not a flag to fix — it is a caller that took the
/// wrong branch, and whose own gateway form still says non-streaming. Setting
/// it here would send a request its caller does not think it sent, and leave
/// the mistake to surface somewhere with less context. `render_request` puts
/// the gateway form's answer on the body; this asserts that it did.
///
/// `openai_compat`'s transport checks the same invariant the same way.
fn request(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    body: &CreateMessageRequest,
) -> reqwest::RequestBuilder {
    http.post(format!("{}/messages", base_url.trim_end_matches('/')))
        .header("x-api-key", api_key)
        .header("anthropic-version", API_VERSION)
        .json(body)
}

/// The events of one streamed reply, read off the socket as they arrive.
///
/// [`SseFrames`] does the framing; this adds the socket, the decode, and the
/// judgement the framing cannot make — where the stream ends. An inherent
/// `async fn next` rather than a [`futures::Stream`] implementation: every
/// caller here is a loop, and a loop needs no combinators, no pinning, and no
/// dependency.
#[derive(Debug)]
#[allow(dead_code)] // staged: read once the stream conversions land
pub(crate) struct EventStream {
    response: reqwest::Response,
    provider: &'static str,
    frames: SseFrames,
    /// The socket is spent — closed, or failed.
    ended: bool,
    /// A terminal event arrived, so the reply is WHOLE. This is the state
    /// [`SseFrames::done`] holds for a protocol with a sentinel; here it can
    /// only be reached by decoding, so it lives on this side of the decode.
    complete: bool,
    /// The socket closed with no terminal event, so the reply is incomplete
    /// and the caller is owed one error saying so.
    ///
    /// Not reported the instant it is noticed: an event may still be sitting
    /// in [`SseFrames`] unflushed, and that event is the caller's before the
    /// bad news is.
    truncated: bool,
    /// The longest gap between reads this stream will sit through. `None`
    /// waits forever, bounded only by the client's total deadline.
    read_timeout: Option<Duration>,
}

/// Whether this event ends the stream.
///
/// `message_stop` is the documented ending. `error` is the OTHER one, and
/// missing it would be the more damaging omission: Anthropic sends errors
/// mid-stream after a 200 — an overload during generation — and nothing
/// follows one. A reader that kept waiting would sit until the socket closed
/// and then report a truncation, burying the reason the provider actually
/// gave.
#[allow(dead_code)] // staged: read once the stream conversions land
fn ends_the_stream(event: &MessageStreamEvent) -> bool {
    matches!(
        event,
        MessageStreamEvent::MessageStop(_) | MessageStreamEvent::Error(_)
    )
}

#[allow(dead_code)] // staged: read once the stream conversions land
impl EventStream {
    /// The next event, or `None` once the stream ends.
    ///
    /// `None` means the reply is WHOLE: a terminal event arrived, or an error
    /// item already said why it will not. A socket that closed early is
    /// neither, and comes back as [`Error::StreamTruncated`] rather than as
    /// the same `None` a finished stream gives — otherwise a truncated answer
    /// is indistinguishable from a complete one at every surface above this.
    ///
    /// `None` is terminal either way: after `message_stop`, a spent socket, or
    /// any error, this keeps returning `None` rather than reading a socket
    /// with nothing left to say.
    pub(crate) async fn next(&mut self) -> Option<Result<MessageStreamEvent>> {
        loop {
            if self.ended || self.complete {
                return None;
            }
            // Checked AFTER `complete`: a body whose last event is
            // `message_stop` without its blank line ends cleanly, and the
            // flush below is what discovers that.
            if self.truncated {
                self.ended = true;
                return Some(Err(Error::StreamTruncated {
                    provider: self.provider,
                }));
            }
            // Everything already buffered first; only read when it runs dry.
            match self.frames.next_payload() {
                Ok(Some(payload)) => return Some(self.take(&payload)),
                Ok(None) => {}
                Err(e) => {
                    self.ended = true;
                    return Some(Err(self.decode_error(&e.to_string())));
                }
            }
            match self.buffer_more().await {
                Ok(true) => {}
                // The socket closed without a terminal event. One event may
                // still be accumulated — a body that omits its final blank
                // line — and it is handed over first; the truncation is
                // reported on the next call, from the flag above.
                Ok(false) => {
                    self.truncated = true;
                    if let Some(payload) = self.frames.flush() {
                        return Some(self.take(&payload));
                    }
                }
                Err(e) => {
                    self.ended = true;
                    return Some(Err(e));
                }
            }
        }
    }

    /// Decodes a payload and records what it means for the stream's lifetime.
    ///
    /// An undecodable event is terminal — the one path that can hand back an
    /// error with the socket still open and more events buffered behind it, so
    /// the stream ends here rather than resuming past something the caller was
    /// already told about.
    fn take(&mut self, payload: &str) -> Result<MessageStreamEvent> {
        let decoded = self.decode(payload);
        match &decoded {
            Ok(event) if ends_the_stream(event) => self.complete = true,
            Err(_) => self.ended = true,
            Ok(_) => {}
        }
        decoded
    }

    /// Reads the next bytes off the socket into [`SseFrames`], answering
    /// whether the socket is still open — `false` is EOF, and the only thing
    /// the caller does differently with it.
    ///
    /// Owns the read timeout because this is the single `await` a stream can
    /// hang on. A stall and a broken socket stay DISTINCT failures: one means
    /// the provider went quiet on a connection that is still up, the other
    /// that the connection went away, and only the first is answered by
    /// raising a limit.
    async fn buffer_more(&mut self) -> Result<bool> {
        let read = match self.read_timeout {
            Some(limit) => tokio::time::timeout(limit, self.response.chunk())
                .await
                .map_err(|_| Error::StreamStalled {
                    provider: self.provider,
                    timeout: limit,
                })?,
            None => self.response.chunk().await,
        };
        match read.map_err(|e| network_error(self.provider, e))? {
            Some(bytes) => {
                self.frames.push(&bytes);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn decode(&self, payload: &str) -> Result<MessageStreamEvent> {
        serde_json::from_str(payload).map_err(|e| self.decode_error(&e.to_string()))
    }

    /// An event that will not decode is nobody's to reinterpret — the same
    /// judgement [`post`] makes about a 2xx body.
    fn decode_error(&self, message: &str) -> Error {
        Error::Decode {
            provider: self.provider,
            message: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn reply() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
    }

    async fn post_to(server: &MockServer) -> Result<Outcome> {
        post(
            &reqwest::Client::new(),
            "anthropic",
            &server.uri(),
            "sk-ant-demo",
            &CreateMessageRequest::default(),
        )
        .await
    }

    /// The contract this module exists to state: a non-2xx is DATA, not an
    /// `Err`. Mapping it to an [`Error`] belongs to the caller.
    #[tokio::test]
    async fn a_non_2xx_comes_back_as_status_with_the_body_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(529).set_body_string("{\"type\":\"error\"}"))
            .mount(&server)
            .await;

        match post_to(&server).await.unwrap() {
            Outcome::Status { status, body, .. } => {
                assert_eq!(status, 529, "Anthropic's own overload status");
                assert_eq!(body, "{\"type\":\"error\"}");
            }
            Outcome::Ok(_) => panic!("a 529 decoded as success"),
        }
    }

    /// ...whereas a 2xx that will not decode is nobody's to reinterpret.
    #[tokio::test]
    async fn a_2xx_that_will_not_decode_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        assert!(matches!(
            post_to(&server).await,
            Err(Error::Decode {
                provider: "anthropic",
                ..
            })
        ));
    }

    /// The path and both headers live only here, so they are only asserted
    /// here: the mock matches nothing else and would 404.
    ///
    /// `anthropic-version` is not decoration — a request without it is
    /// rejected outright — so it is pinned alongside the key.
    #[tokio::test]
    async fn posts_to_messages_with_an_api_key_and_a_version_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "sk-ant-demo"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply()))
            .expect(1)
            .mount(&server)
            .await;

        assert!(matches!(post_to(&server).await.unwrap(), Outcome::Ok(_)));
    }

    /// A bearer token is what the OTHER protocol sends, and sending one here
    /// would authenticate against nothing.
    #[tokio::test]
    async fn the_api_key_does_not_go_in_an_authorization_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply()))
            .mount(&server)
            .await;

        post_to(&server).await.unwrap();
        let sent = &server.received_requests().await.unwrap()[0];
        assert!(sent.headers.get("authorization").is_none());
    }

    /// A trailing slash on the base URL must not double up in the path.
    #[tokio::test]
    async fn a_trailing_slash_on_the_base_url_is_trimmed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .expect(1)
            .mount(&server)
            .await;

        // Decode is the expected outcome; reaching the mock at all is the point.
        let err = post(
            &reqwest::Client::new(),
            "anthropic",
            &format!("{}/", server.uri()),
            "sk-ant-demo",
            &CreateMessageRequest::default(),
        )
        .await;
        assert!(matches!(err, Err(Error::Decode { .. })), "{err:?}");
    }

    // -----------------------------------------------------------------------
    // Streaming
    //
    // The SSE framing is shared and tested at `crate::http::SseFrames`. What is
    // protocol-specific — and tested here — is where this protocol's stream
    // ENDS, since it has no sentinel payload to match and must decode to find
    // out.
    // -----------------------------------------------------------------------

    /// A whole minimal transcript, in Anthropic's own framing: named events,
    /// a keepalive, and `message_stop` in place of a `[DONE]` sentinel.
    const SSE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",",
        "\"role\":\"assistant\",\"content\":[],\"model\":\"claude-opus-5\",\"stop_reason\":null,",
        "\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,",
        "\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    /// The body every streamed call arrives with — `render_request` puts the
    /// gateway form's answer here, and `post_stream` refuses anything else.
    fn streamed() -> CreateMessageRequest {
        CreateMessageRequest {
            stream: Some(true),
            ..Default::default()
        }
    }

    async fn stream_from(server: &MockServer) -> Result<StreamOutcome> {
        post_stream(
            &reqwest::Client::new(),
            "anthropic",
            &server.uri(),
            "sk-ant-demo",
            &streamed(),
            // Unbounded, which is the default and what every test but the
            // stall one below wants: a mock answers instantly, so a limit
            // here could only ever fire spuriously.
            None,
        )
        .await
    }

    async fn stream_of(body: String) -> EventStream {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let StreamOutcome::Ok(stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        // The mock server must outlive the socket, so it is leaked rather than
        // dropped at the end of this helper.
        std::mem::forget(server);
        stream
    }

    async fn collect(mut stream: EventStream) -> Vec<Result<MessageStreamEvent>> {
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    /// Every event reaches the caller, `message_stop` included — unlike the
    /// `[DONE]` sentinel it replaces, it is a real event carrying a real
    /// position in the stream, and the layer above needs to see it.
    #[tokio::test]
    async fn a_streamed_body_yields_every_event_including_the_last() {
        let events = collect(stream_of(SSE.to_string()).await).await;
        let kinds: Vec<_> = events
            .iter()
            .map(|event| match event.as_ref().unwrap() {
                MessageStreamEvent::MessageStart(_) => "message_start",
                MessageStreamEvent::Ping(_) => "ping",
                MessageStreamEvent::ContentBlockDelta(_) => "content_block_delta",
                MessageStreamEvent::MessageStop(_) => "message_stop",
                other => panic!("unexpected event: {other:?}"),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "message_start",
                "ping",
                "content_block_delta",
                "message_stop"
            ],
            "the `event:` lines must not disturb the payloads"
        );
    }

    /// `None` is terminal: a spent stream keeps saying so rather than reading
    /// a socket that has nothing left.
    #[tokio::test]
    async fn a_finished_stream_stays_finished() {
        let mut stream = stream_of(SSE.to_string()).await;
        while stream.next().await.is_some() {}
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }

    /// Nothing after the terminal event is the caller's, which is the
    /// behaviour a sentinel would give for free and this protocol has to
    /// decide for itself.
    #[tokio::test]
    async fn nothing_after_message_stop_reaches_the_caller() {
        let events = collect(stream_of(format!("{SSE}{SSE}")).await).await;
        assert_eq!(events.len(), 4, "the second transcript must not be read");
    }

    /// A stream that stops without a terminal event is TRUNCATED, and says so.
    ///
    /// The bug this pins is a silent one: without the distinction, a
    /// connection that dropped mid-answer ends in exactly the `None` a
    /// finished stream ends in, so every surface above reports a half-written
    /// reply as a complete one.
    #[tokio::test]
    async fn a_socket_that_closes_before_message_stop_is_an_error() {
        let body = SSE.split("event: message_stop").next().unwrap().to_string();
        let mut stream = stream_of(body).await;
        // Everything that DID arrive still reaches the caller first.
        for _ in 0..3 {
            assert!(stream.next().await.unwrap().is_ok());
        }
        assert!(
            matches!(
                stream.next().await,
                Some(Err(Error::StreamTruncated {
                    provider: "anthropic"
                }))
            ),
            "a stream that stopped early must not end like one that finished"
        );
        assert!(stream.next().await.is_none(), "the error is terminal");
    }

    /// ...but a body whose last event is `message_stop`, missing only the
    /// blank line that would dispatch it, FINISHED. The socket closing is what
    /// dispatches it, so the flush has to be read before truncation is
    /// declared.
    #[tokio::test]
    async fn a_terminal_event_without_its_trailing_blank_line_still_ends_cleanly() {
        let mut stream = stream_of(SSE.strip_suffix('\n').unwrap().to_string()).await;
        for _ in 0..4 {
            assert!(stream.next().await.unwrap().is_ok());
        }
        assert!(stream.next().await.is_none(), "message_stop did arrive");
    }

    /// A mid-stream `error` ends the stream too, and reaches the caller as the
    /// event it is.
    ///
    /// This is the case a 200 does not rule out — an overload during
    /// generation — and treating it as an ordinary event would leave the
    /// reader waiting for a `message_stop` that is never coming, so the
    /// provider's stated reason would be replaced by a truncation.
    #[tokio::test]
    async fn a_mid_stream_error_event_ends_the_stream() {
        let error = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":",
            "{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        );
        let body = SSE.split("event: message_stop").next().unwrap();
        let events = collect(stream_of(format!("{body}{error}{SSE}")).await).await;
        assert_eq!(events.len(), 4, "nothing after the error is read");
        let MessageStreamEvent::Error(event) = events[3].as_ref().unwrap() else {
            panic!("the error did not reach the caller as an event");
        };
        assert_eq!(event.error.r#type, "overloaded_error");
    }

    /// A malformed TERMINAL event is a decode failure, not a truncation.
    ///
    /// This is the consequence that makes the reader's `Unknown` arm mean "a
    /// tag warpllm does not know" rather than "anything that did not parse".
    /// Were `{"type": "error"}` with no `error` object read as an unknown
    /// event, [`ends_the_stream`] would not recognize it, this stream would
    /// wait for a `message_stop` that is never coming, and the socket closing
    /// behind it would be reported as [`Error::StreamTruncated`] — a
    /// misleading answer, since nothing was truncated and the provider did say
    /// something.
    ///
    /// Both malformed shapes are checked, because the framing makes them
    /// indistinguishable to this reader: it ignores the `event:` line, so an
    /// `event: error` is only ever recognized by what is inside its `data:`.
    /// A payload that lost its `type` is therefore not "an error event missing
    /// a field" — it is a payload with nothing to identify it at all.
    #[tokio::test]
    async fn a_malformed_terminal_event_is_a_decode_error_not_a_truncation() {
        let malformed = [
            // A known tag whose body does not parse.
            "{\"type\":\"error\"}",
            // No discriminator at all, under an `event: error` line the
            // framing does not read.
            "{\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
        ];
        for payload in malformed {
            let body = SSE.split("event: message_stop").next().unwrap();
            let mut stream = stream_of(format!("{body}event: error\ndata: {payload}\n\n")).await;
            for _ in 0..3 {
                assert!(stream.next().await.unwrap().is_ok());
            }
            assert!(
                matches!(
                    stream.next().await,
                    Some(Err(Error::Decode {
                        provider: "anthropic",
                        ..
                    }))
                ),
                "{payload} passed as an unknown event instead of failing to decode"
            );
            assert!(stream.next().await.is_none(), "the error is terminal");
        }
    }

    /// The same contract [`post`] states, on the streaming path: a non-2xx is
    /// DATA, read to the end and handed over verbatim.
    #[tokio::test]
    async fn a_non_2xx_comes_back_as_status_without_streaming() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_string("slow down"),
            )
            .mount(&server)
            .await;

        match stream_from(&server).await.unwrap() {
            StreamOutcome::Status {
                status,
                body,
                retry_after,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(body, "slow down");
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
            }
            StreamOutcome::Ok(_) => panic!("a 429 opened a stream"),
        }
    }

    /// An event that will not decode ends the stream as an error, rather than
    /// being skipped as though the provider had sent nothing.
    ///
    /// The bad event is deliberately NOT the last one. A body that ended right
    /// after it would return `None` next whatever this code did, because the
    /// socket closed — proving the SOCKET ended, not the error.
    #[tokio::test]
    async fn an_undecodable_event_is_an_error_that_ends_the_stream() {
        let mut stream = stream_of(format!("data: not json\n\n{SSE}")).await;
        assert!(matches!(
            stream.next().await,
            Some(Err(Error::Decode {
                provider: "anthropic",
                ..
            }))
        ));
        assert!(
            stream.next().await.is_none(),
            "the stream resumed past an error it had already reported"
        );
    }

    /// A stream that goes QUIET — no bytes, and no close either — is ended by
    /// `read_timeout` and nothing else.
    ///
    /// This is the failure the client's total deadline answers only by
    /// outliving: the connection is up, the provider is simply not talking. It
    /// needs a raw socket because a mock server cannot produce it — wiremock's
    /// `set_delay` holds back the whole response, which stalls the request
    /// before its headers and never reaches the read this bounds.
    ///
    /// The event sent first is a `ping`, which is a real Anthropic keepalive
    /// and not an ending — so this also pins that a keepalive is never what
    /// ends a stream.
    #[tokio::test]
    async fn a_stream_that_goes_quiet_is_ended_at_the_read_timeout() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        const LIMIT: Duration = Duration::from_millis(100);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = [0u8; 2048];
            assert!(socket.read(&mut head).await.unwrap() > 0, "no request");
            let event = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                         transfer-encoding: chunked\r\n\r\n{:x}\r\n{event}\r\n",
                        event.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            // ...and then nothing at all, with the socket still open.
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let StreamOutcome::Ok(mut stream) = post_stream(
            &reqwest::Client::new(),
            "anthropic",
            &format!("http://{addr}"),
            "sk-ant-demo",
            &streamed(),
            Some(LIMIT),
        )
        .await
        .unwrap() else {
            panic!("a 200 did not stream");
        };

        assert!(matches!(
            stream.next().await,
            Some(Ok(MessageStreamEvent::Ping(_)))
        ));
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            matches!(
                &error,
                Error::StreamStalled {
                    provider: "anthropic",
                    timeout,
                } if *timeout == LIMIT
            ),
            "a quiet socket is a stall, not a truncation or a network error: {error:?}"
        );
        assert!(stream.next().await.is_none(), "the error is terminal");
    }

    /// The body actually goes out as JSON, `stream: true` included — the flag
    /// is what makes the reply an event stream at all.
    #[tokio::test]
    async fn the_request_body_is_sent_as_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(body_partial_json(
                serde_json::json!({"model": "claude-opus-5", "max_tokens": 7, "stream": true}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string(SSE))
            .expect(1)
            .mount(&server)
            .await;

        let outcome = post_stream(
            &reqwest::Client::new(),
            "anthropic",
            &server.uri(),
            "sk-ant-demo",
            &CreateMessageRequest {
                model: "claude-opus-5".to_string(),
                max_tokens: 7,
                stream: Some(true),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(matches!(outcome, StreamOutcome::Ok(_)));
    }

    /// A body that disagrees with the function called for it is REFUSED, not
    /// corrected.
    ///
    /// `stream: false` here means a caller took the wrong branch, and its own
    /// gateway form still says non-streaming. Setting the flag on its behalf
    /// would send a request the caller does not think it sent; the failure it
    /// was heading for otherwise is a complete reply misreported as a
    /// truncation.
    #[tokio::test]
    async fn a_streamed_call_refuses_a_body_that_does_not_ask_for_a_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(SSE))
            .expect(0)
            .mount(&server)
            .await;

        for stream in [None, Some(false)] {
            let refused = post_stream(
                &reqwest::Client::new(),
                "anthropic",
                &server.uri(),
                "sk-ant-demo",
                &CreateMessageRequest {
                    stream,
                    ..Default::default()
                },
                None,
            )
            .await;
            assert!(
                matches!(refused, Err(Error::InvalidInput(_))),
                "{stream:?} opened a stream: {refused:?}"
            );
        }
    }

    /// ...and the mirror: a whole reply cannot carry events, so asking for
    /// them here is refused rather than silently unasked.
    #[tokio::test]
    async fn an_unstreamed_call_refuses_a_body_that_asks_for_a_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply()))
            .expect(0)
            .mount(&server)
            .await;

        let refused = post(
            &reqwest::Client::new(),
            "anthropic",
            &server.uri(),
            "sk-ant-demo",
            &CreateMessageRequest {
                stream: Some(true),
                ..Default::default()
            },
        )
        .await;
        assert!(
            matches!(refused, Err(Error::InvalidInput(_))),
            "{refused:?}"
        );
    }
}
