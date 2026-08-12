//! The HTTP binding for `POST /chat/completions`: the one place the
//! [`Api::ChatCompletions`](crate::Api) → URL path mapping physically lives,
//! since the module path spells the API's name rather than its route.

use std::time::Duration;

use crate::error::{Error, Result};
use crate::http::{network_error, read_response};
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};

/// The payload that ends an OpenAI-compatible stream. Not JSON, and not a
/// chunk — the one `data:` value that has to be recognized rather than parsed.
const DONE: &str = "[DONE]";

/// What the upstream said. A non-2xx is NOT an [`Err`] here: which [`Error`] a
/// given status and body becomes is the caller's to decide: a provider may
/// envelope its errors differently from the protocol default, and deciding here
/// would mean `protocol` reaching into the conversion layer to find out, which
/// is the dependency this module exists without. `Err` is reserved for failures
/// nothing could reinterpret — the request never completing, or a 2xx body that
/// will not decode.
///
/// `large_enum_variant` is allowed rather than fixed: boxing the success
/// variant would add an allocation to every successful request to shrink a
/// value that is constructed once, moved once, and destructured immediately.
/// The lint's premise — many of these held at once — never happens.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Outcome {
    Ok(CreateChatCompletionResponse),
    /// The status, raw body, and header evidence, verbatim, for the caller
    /// to map.
    ///
    /// The headers travel with the body because this is the LAST place they
    /// exist: `Retry-After` and the upstream request id appear nowhere in
    /// any error envelope, so a caller handed only a status and a body
    /// cannot answer how long to wait or what to quote to the provider.
    Status {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
        request_id: Option<String>,
    },
}

pub(crate) async fn post(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateChatCompletionRequest,
) -> Result<Outcome> {
    let response = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(body)
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

/// [`Outcome`] for a streamed request. Same contract: a non-2xx is DATA, and
/// only the caller decides which [`Error`] it becomes.
///
/// The success arm holds an open socket rather than a decoded body, which is
/// the whole difference between this and [`Outcome`]: nothing has been read
/// past the headers, so the status decision is made before the first chunk and
/// never again.
#[derive(Debug)]
pub(crate) enum StreamOutcome {
    Ok(ChunkStream),
    /// The status, raw body, and header evidence, verbatim, for the caller to
    /// map — see [`Outcome::Status`].
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
/// A non-2xx body is read to the end here, exactly as [`post`] does: an error
/// reply is small, complete, and worth nothing streamed.
pub(crate) async fn post_stream(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateChatCompletionRequest,
) -> Result<StreamOutcome> {
    let response = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(body)
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

    Ok(StreamOutcome::Ok(ChunkStream {
        response,
        provider,
        frames: Frames::default(),
        ended: false,
    }))
}

/// Server-sent event framing, with no socket in it: bytes in, payloads out.
///
/// Separate from [`ChunkStream`] because this is the fiddly half and the half
/// worth testing exhaustively — a read boundary falls wherever TCP puts it,
/// which is routinely mid-line and can be mid-character. Keeping it free of
/// I/O means every edge below is a plain function call in a test.
///
/// It reads the subset OpenAI-compatible providers actually send:
///
/// - `data:` lines accumulate into the current event, joined by newlines when
///   a provider splits one across several (the specification allows it; none
///   of them do it today);
/// - a blank line dispatches the accumulated event;
/// - `:` comment lines — the usual keepalive — and every other SSE field
///   (`event:`, `id:`, `retry:`) are ignored;
/// - the [`DONE`] payload ends the stream rather than becoming an event.
#[derive(Debug, Default)]
struct Frames {
    /// Read but not yet consumed as complete lines.
    buffer: Vec<u8>,
    /// The `data:` payload of the event being accumulated.
    event: String,
    done: bool,
}

impl Frames {
    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next dispatched payload, or `None` when the bytes so far do not
    /// contain a complete event — which is the signal to go read more.
    ///
    /// Lines are decoded as UTF-8 only once complete, so a multi-byte
    /// character split across two reads is rejoined rather than mangled.
    // `std::result::Result` spelled out: the crate's own `Result` alias fixes
    // the error type, and this one is a UTF-8 failure the caller labels.
    fn next_payload(&mut self) -> std::result::Result<Option<String>, std::str::Utf8Error> {
        while !self.done {
            let Some(line) = self.take_line() else {
                return Ok(None);
            };
            let line = std::str::from_utf8(&line)?;
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                if let Some(payload) = self.dispatch() {
                    return Ok(Some(payload));
                }
            } else if let Some(payload) = line.strip_prefix("data:") {
                if !self.event.is_empty() {
                    self.event.push('\n');
                }
                // One optional leading space belongs to the framing.
                self.event
                    .push_str(payload.strip_prefix(' ').unwrap_or(payload));
            }
        }
        Ok(None)
    }

    /// The event accumulated so far, for a socket that closed without sending
    /// its final blank line — providers do end streams that way.
    fn flush(&mut self) -> Option<String> {
        self.dispatch()
    }

    /// The next complete line, `\n` removed, or `None` when the buffer holds
    /// only a partial one.
    fn take_line(&mut self) -> Option<Vec<u8>> {
        let end = self.buffer.iter().position(|byte| *byte == b'\n')?;
        let mut line: Vec<u8> = self.buffer.drain(..=end).collect();
        line.pop();
        Some(line)
    }

    fn dispatch(&mut self) -> Option<String> {
        let payload = std::mem::take(&mut self.event);
        let payload = payload.trim();
        if payload.is_empty() {
            return None;
        }
        if payload == DONE {
            self.done = true;
            return None;
        }
        Some(payload.to_string())
    }
}

/// The chunks of one streamed reply, read off the socket as they arrive.
///
/// [`Frames`] does the framing; this adds the socket and the decode. An
/// inherent `async fn next` rather than a [`futures::Stream`] implementation:
/// every caller here is a loop, and a loop needs no combinators, no pinning,
/// and no dependency.
#[derive(Debug)]
pub(crate) struct ChunkStream {
    response: reqwest::Response,
    provider: &'static str,
    frames: Frames,
    /// The socket is spent — closed, or failed. Distinct from
    /// [`Frames::done`], which means the sentinel arrived.
    ended: bool,
}

impl ChunkStream {
    /// The next chunk, or `None` once the stream ends.
    ///
    /// `None` is terminal: after `[DONE]`, a closed socket, or any error, this
    /// keeps returning `None` rather than reading a socket with nothing left
    /// to say.
    pub(crate) async fn next(&mut self) -> Option<Result<CreateChatCompletionStreamResponse>> {
        loop {
            if self.ended || self.frames.done {
                return None;
            }
            // Everything already buffered first; only read when it runs dry.
            match self.frames.next_payload() {
                Ok(Some(payload)) => {
                    let decoded = self.decode(&payload);
                    // The one path that can hand back an error with the socket
                    // still open and more events already buffered behind it.
                    // An error item is terminal, so it ends the stream here
                    // rather than letting the next call resume past it.
                    if decoded.is_err() {
                        self.ended = true;
                    }
                    return Some(decoded);
                }
                // The sentinel ends the stream where it sits. Falling through
                // to read again would block on an upstream that holds the
                // response open after sending it — and the sentinel, not the
                // socket, is what says a stream is over.
                Ok(None) if self.frames.done => return None,
                Ok(None) => {}
                Err(e) => {
                    self.ended = true;
                    return Some(Err(self.decode_error(&e.to_string())));
                }
            }
            match self.response.chunk().await {
                Ok(Some(bytes)) => self.frames.push(&bytes),
                Ok(None) => {
                    self.ended = true;
                    return self.frames.flush().map(|payload| self.decode(&payload));
                }
                Err(e) => {
                    self.ended = true;
                    return Some(Err(network_error(self.provider, e)));
                }
            }
        }
    }

    fn decode(&self, payload: &str) -> Result<CreateChatCompletionStreamResponse> {
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
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn post_to(server: &MockServer) -> Result<Outcome> {
        post(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            "sk-demo",
            &CreateChatCompletionRequest::default(),
        )
        .await
    }

    /// The contract this module exists to state: a non-2xx is DATA, not an
    /// `Err`. Mapping it to an [`Error`] belongs to the caller, since a
    /// provider may envelope its errors differently from the protocol default.
    #[tokio::test]
    async fn a_non_2xx_comes_back_as_status_with_the_body_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .mount(&server)
            .await;

        match post_to(&server).await.unwrap() {
            Outcome::Status { status, body, .. } => {
                assert_eq!(status, 429);
                assert_eq!(body, "slow down", "the body must reach the caller unread");
            }
            Outcome::Ok(_) => panic!("a 429 decoded as success"),
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
                provider: "demo",
                ..
            })
        ));
    }

    /// The URL suffix and bearer scheme live only here, so they are only
    /// asserted here: the mock matches nothing else and would 404.
    #[tokio::test]
    async fn posts_to_chat_completions_with_a_bearer_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-demo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "demo",
                "choices": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        assert!(matches!(post_to(&server).await.unwrap(), Outcome::Ok(_)));
    }

    // -----------------------------------------------------------------------
    // Framing
    // -----------------------------------------------------------------------

    /// Every payload the framing yields from one push of bytes.
    fn payloads(chunks: &[&str]) -> Vec<String> {
        let mut frames = Frames::default();
        let mut out = Vec::new();
        for bytes in chunks {
            frames.push(bytes.as_bytes());
            while let Some(payload) = frames.next_payload().unwrap() {
                out.push(payload);
            }
        }
        out.extend(frames.flush());
        out
    }

    /// The ordinary case, one whole body at a time.
    #[test]
    fn data_lines_dispatch_on_a_blank_line_and_done_ends_the_stream() {
        assert_eq!(
            payloads(&["data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n"]),
            ["{\"a\":1}", "{\"a\":2}"],
            "the sentinel is the stream ending, never an event"
        );
    }

    /// The case no fixture can produce: a read boundary lands mid-line, and
    /// mid-character. Neither may lose or mangle a byte.
    #[test]
    fn an_event_split_across_reads_is_rejoined() {
        assert_eq!(payloads(&["data: {\"a\":", "1}\n\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data: {\"a\":1}", "\n", "\n"]), ["{\"a\":1}"]);
        // "일" is three bytes; the split falls inside it.
        let text = "data: {\"a\":\"일\"}\n\n".as_bytes();
        let (head, tail) = text.split_at(14);
        let mut frames = Frames::default();
        frames.push(head);
        assert_eq!(frames.next_payload().unwrap(), None, "a partial character");
        frames.push(tail);
        assert_eq!(
            frames.next_payload().unwrap().as_deref(),
            Some("{\"a\":\"일\"}")
        );
    }

    /// Keepalive comments and the SSE fields warpllm does not read must not
    /// become events, and must not disturb the one being accumulated.
    #[test]
    fn comments_and_other_fields_are_ignored() {
        assert_eq!(
            payloads(&[": ping\nevent: message\nid: 7\nretry: 100\ndata: {\"a\":1}\n\n"]),
            ["{\"a\":1}"]
        );
        assert_eq!(payloads(&[": ping\n\n: ping\n\n"]), [] as [String; 0]);
    }

    /// CRLF and the optional leading space are framing, not payload.
    #[test]
    fn crlf_and_the_optional_space_are_stripped() {
        assert_eq!(payloads(&["data: {\"a\":1}\r\n\r\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data:{\"a\":1}\n\n"]), ["{\"a\":1}"]);
        // Only ONE space belongs to the framing; the rest is payload, and
        // `trim` on dispatch is what keeps it decodable either way.
        assert_eq!(payloads(&["data:  {\"a\":1}\n\n"]), ["{\"a\":1}"]);
    }

    /// The specification allows one event's data to span several lines; no
    /// provider does it today, so this pins the behaviour rather than a bug.
    #[test]
    fn multi_line_data_joins_with_newlines() {
        assert_eq!(payloads(&["data: {\"a\":\ndata: 1}\n\n"]), ["{\"a\":\n1}"]);
    }

    /// The sentinel leaves the framing in the one state a reader must not
    /// mistake for "send me more bytes": `next_payload` says `Ok(None)` to
    /// both, and only [`Frames::done`] tells them apart.
    ///
    /// That ambiguity is why [`ChunkStream::next`] rechecks `done` before
    /// awaiting the socket. Without the recheck an upstream that holds the
    /// response open after `[DONE]` blocks the caller forever — a hang, so the
    /// symptom is untestable, but the state that causes it is exactly this.
    #[test]
    fn the_sentinel_leaves_no_payload_and_a_done_flag() {
        let mut frames = Frames::default();
        frames.push(b"data: [DONE]\n\n");
        assert_eq!(frames.next_payload().unwrap(), None);
        assert!(frames.done, "only this distinguishes ended from starved");

        let mut starved = Frames::default();
        starved.push(b"data: {\"a\":1}");
        assert_eq!(starved.next_payload().unwrap(), None);
        assert!(!starved.done);
    }

    /// A body that ends without its final blank line still has an event.
    #[test]
    fn a_truncated_final_event_is_flushed() {
        assert_eq!(payloads(&["data: {\"a\":1}\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data: [DONE]\n"]), [] as [String; 0]);
    }

    /// Invalid UTF-8 is reported rather than replaced: a lossy decode would
    /// hand the caller a chunk the provider never sent.
    #[test]
    fn invalid_utf8_in_a_line_is_an_error() {
        let mut frames = Frames::default();
        frames.push(b"data: \xff\xfe\n\n");
        assert!(frames.next_payload().is_err());
    }

    // -----------------------------------------------------------------------
    // Streaming transport
    // -----------------------------------------------------------------------

    const SSE: &str = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",",
        "\"created\":1700000000,\"model\":\"demo\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        ": keepalive\n\n",
        "data: [DONE]\n\n",
    );

    async fn stream_from(server: &MockServer) -> Result<StreamOutcome> {
        post_stream(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            "sk-demo",
            &CreateChatCompletionRequest::default(),
        )
        .await
    }

    async fn collect(mut stream: ChunkStream) -> Vec<Result<CreateChatCompletionStreamResponse>> {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }
        chunks
    }

    #[tokio::test]
    async fn a_streamed_body_yields_its_chunks_and_stops_at_done() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-demo"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(SSE),
            )
            .expect(1)
            .mount(&server)
            .await;

        let StreamOutcome::Ok(stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        let chunks = collect(stream).await;
        assert_eq!(chunks.len(), 1, "the sentinel must not become a chunk");
        assert_eq!(chunks[0].as_ref().unwrap().id, "chatcmpl-1");
    }

    /// `None` is terminal: a spent stream keeps saying so rather than reading
    /// a socket that has nothing left.
    #[tokio::test]
    async fn a_finished_stream_stays_finished() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(SSE))
            .mount(&server)
            .await;

        let StreamOutcome::Ok(mut stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        assert!(stream.next().await.is_some());
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
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

    /// ...and an event that will not decode ends the stream as an error,
    /// rather than being skipped as though the provider had sent nothing.
    ///
    /// The bad event is deliberately NOT the last one. A body that ended right
    /// after it would return `None` next whatever this code did, because the
    /// socket closed — proving the SOCKET ended, not the error. A well-formed
    /// event queued behind it is what makes the terminal-error contract
    /// testable at all: resuming would hand a caller a chunk after it had
    /// already been told the stream failed.
    #[tokio::test]
    async fn an_undecodable_event_is_an_error_that_ends_the_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(format!("data: not json\n\n{SSE}")),
            )
            .mount(&server)
            .await;

        let StreamOutcome::Ok(mut stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        assert!(matches!(
            stream.next().await,
            Some(Err(Error::Decode {
                provider: "demo",
                ..
            }))
        ));
        assert!(
            stream.next().await.is_none(),
            "the stream resumed past an error it had already reported"
        );
    }

    /// An event queued behind the sentinel is not the caller's: `[DONE]` ends
    /// the stream where it sits, and nothing after it is read.
    #[tokio::test]
    async fn nothing_after_the_sentinel_reaches_the_caller() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(format!("data: [DONE]\n\n{SSE}")),
            )
            .mount(&server)
            .await;

        let StreamOutcome::Ok(mut stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }

    /// A trailing slash on the base URL must not double up in the path.
    #[tokio::test]
    async fn a_trailing_slash_on_the_base_url_is_trimmed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .expect(1)
            .mount(&server)
            .await;

        // Decode is the expected outcome; reaching the mock at all is the point.
        let err = post(
            &reqwest::Client::new(),
            "demo",
            &format!("{}/", server.uri()),
            "sk-demo",
            &CreateChatCompletionRequest::default(),
        )
        .await;
        assert!(matches!(err, Err(Error::Decode { .. })), "{err:?}");
    }
}
