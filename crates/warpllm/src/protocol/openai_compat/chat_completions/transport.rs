//! The HTTP binding for `POST /chat/completions`: the one place the
//! [`Api::ChatCompletions`](crate::Api) → URL path mapping physically lives,
//! since the module path spells the API's name rather than its route.

use std::time::Duration;

use crate::auth::Authenticator;
use crate::error::{Error, Result};
use crate::http::{SseFrames, network_error, read_response};
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

/// Sends a request and reads the whole reply.
///
/// REQUIRES a body that does not ask for a stream, and says so rather than
/// quietly making it true — see [`post_stream`] for why the check is here at
/// all, and why it refuses rather than corrects.
pub(crate) async fn post(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: Option<&Authenticator>,
    body: &CreateChatCompletionRequest,
) -> Result<Outcome> {
    if body.stream == Some(true) {
        return Err(Error::InvalidInput(
            "stream: true asks for chunks; a whole reply cannot carry them".into(),
        ));
    }
    let response = send(http, provider, base_url, auth, body).await?;

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

/// Sends a `stream: true` request and hands back the chunks, undecoded until
/// asked for.
///
/// REQUIRES a body that already asks for a stream, and REFUSES one that does
/// not rather than setting the flag on its behalf.
///
/// The check exists because getting it wrong fails silently: a body saying
/// `stream: false` would come back as an ordinary JSON completion, which
/// [`ChunkStream`] then reads as SSE — no `data:` line ever dispatches, the
/// socket closes, and the caller is told the reply was TRUNCATED. A complete
/// answer reported as a broken connection, with nothing pointing at the flag.
///
/// It refuses rather than corrects because a body that disagrees with the
/// function called for it is not a flag to fix — it is a caller that took the
/// wrong branch, and whose own gateway form still says non-streaming. Setting
/// the flag here would send a request its caller does not think it sent, which
/// is the same silent rewrite `ensure_renderable` exists to prevent one layer
/// up. Every caller already decides this before arriving:
/// [`Client::chat_completions`](crate::Client::chat_completions) and
/// `JsonChatClient::chat_completions` refuse `stream: true` outright,
/// [`chat_completions_stream`](crate::Client::chat_completions_stream) sets it
/// before ingest, and `render_request` carries the gateway form's answer onto
/// the wire. This is the assertion that they did.
///
/// A non-2xx body is read to the end here, exactly as [`post`] does: an error
/// reply is small, complete, and worth nothing streamed.
///
/// `read_timeout` bounds the GAP between reads once the stream is open. It is
/// applied here rather than on the [`reqwest::Client`] because reqwest's own
/// `read_timeout` is builder-scoped: setting it there would bind every
/// non-streamed request too, or cost a second connection pool to avoid that.
/// The gap is one `await` in [`ChunkStream::next`], so bounding it needs
/// neither.
pub(crate) async fn post_stream(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: Option<&Authenticator>,
    body: &CreateChatCompletionRequest,
    read_timeout: Option<Duration>,
) -> Result<StreamOutcome> {
    if body.stream != Some(true) {
        return Err(Error::InvalidInput(
            "a streamed request must carry stream: true".into(),
        ));
    }
    let response = send(http, provider, base_url, auth, body).await?;

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
        frames: SseFrames::new(Some(DONE)),
        ended: false,
        truncated: false,
        read_timeout,
    }))
}

/// The request both entry points send, differing only in the body's `stream`
/// — built, authenticated, and executed.
///
/// The three steps are one function because the middle one needs the first to
/// have finished. A credential is applied to a BUILT [`reqwest::Request`] rather
/// than a builder, so that a signing scheme (#24) can read the method, the URL
/// and the body it has to sign; `reqwest`'s own `send` would have executed the
/// request before anything could touch it.
///
/// Which header the credential lands in is [`Authenticator`]'s, not this
/// file's. `Authorization: Bearer …` is what every host on the roster reads,
/// but it is a fact about those HOSTS rather than about this wire format —
/// Azure's OpenAI-compatible endpoints read `api-key`, and a self-hosted one
/// (#22) may read neither — so the spelling belongs to the provider and this
/// file states only the path.
/// `auth` is an [`Option`] because a host may genuinely want nothing: the
/// roster's `auth: none`, which is what a self-hosted box on a private network
/// declares. `None` sends the request exactly as built, with no `Authorization`
/// header rather than an empty one — several OpenAI-compatible servers reject a
/// bare `Bearer ` outright, so the two are not the same request.
async fn send(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    auth: Option<&Authenticator>,
    body: &CreateChatCompletionRequest,
) -> Result<reqwest::Response> {
    let request = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .json(body)
        // NOT `Error::Network`: nothing was sent, and nothing was going to be.
        // A URL that will not parse is a warpllm bug or a misconfigured
        // `base_url`, and calling it a network error would send someone
        // looking at the provider's status page.
        .build()
        .map_err(|e| Error::Internal(format!("could not build the {provider} request: {e}")))?;
    let request = match auth {
        Some(auth) => auth.authenticate(request).await?,
        None => request,
    };
    http.execute(request)
        .await
        .map_err(|e| network_error(provider, e))
}

/// The chunks of one streamed reply, read off the socket as they arrive.
///
/// [`SseFrames`] does the framing; this adds the socket and the decode. An
/// inherent `async fn next` rather than a [`futures::Stream`] implementation:
/// every caller here is a loop, and a loop needs no combinators, no pinning,
/// and no dependency.
#[derive(Debug)]
pub(crate) struct ChunkStream {
    response: reqwest::Response,
    provider: &'static str,
    frames: SseFrames,
    /// The socket is spent — closed, or failed. Distinct from
    /// [`SseFrames::done`], which means the sentinel arrived.
    ended: bool,
    /// The socket closed with no sentinel, so the reply is incomplete and the
    /// caller is owed one error saying so.
    ///
    /// Not reported the instant it is noticed: an event may still be sitting
    /// in [`Frames`] unflushed, and that event is the caller's before the bad
    /// news is.
    truncated: bool,
    /// The longest gap between reads this stream will sit through. `None`
    /// waits forever, bounded only by the client's total deadline.
    read_timeout: Option<Duration>,
}

impl ChunkStream {
    /// The next chunk, or `None` once the stream ends.
    ///
    /// `None` means the reply is WHOLE: the sentinel arrived, or an error item
    /// already said why it will not. A socket that closed early is neither, and
    /// comes back as [`Error::StreamTruncated`] rather than as the same `None`
    /// a finished stream gives — otherwise a truncated answer is identical to
    /// a complete one at every surface above this.
    ///
    /// `None` is terminal either way: after `[DONE]`, a spent socket, or any
    /// error, this keeps returning `None` rather than reading a socket with
    /// nothing left to say.
    pub(crate) async fn next(&mut self) -> Option<Result<CreateChatCompletionStreamResponse>> {
        loop {
            if self.ended || self.frames.done {
                return None;
            }
            // Checked AFTER `frames.done`: a body whose last line is the
            // sentinel without its blank line ends cleanly, and the flush
            // below is what discovers that.
            if self.truncated {
                self.ended = true;
                return Some(Err(Error::StreamTruncated {
                    provider: self.provider,
                }));
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
            match self.buffer_more().await {
                Ok(true) => {}
                // The socket closed without a sentinel. One event may still be
                // accumulated — a body that omits its final blank line — and
                // it is handed over first; the truncation is reported on the
                // next call, from the flag above.
                Ok(false) => {
                    self.truncated = true;
                    if let Some(payload) = self.frames.flush() {
                        let decoded = self.decode(&payload);
                        if decoded.is_err() {
                            self.ended = true;
                        }
                        return Some(decoded);
                    }
                }
                Err(e) => {
                    self.ended = true;
                    return Some(Err(e));
                }
            }
        }
    }

    /// Reads the next bytes off the socket into [`Frames`], answering whether
    /// the socket is still open — `false` is EOF, and the only thing the
    /// caller does differently with it.
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

    /// The credential every call below sends. A bearer token, because that is
    /// what an OpenAI-compatible host reads — which provider gets which scheme
    /// is `crate::credentials`', not this module's.
    fn key() -> Authenticator {
        Authenticator::bearer("sk-demo".into())
    }

    async fn post_to(server: &MockServer) -> Result<Outcome> {
        post(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            Some(&key()),
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
    // Streaming transport
    //
    // The SSE framing itself is shared, and tested where it lives, at
    // `crate::http::SseFrames`. What is protocol-specific — and tested here —
    // is what the framing's states MEAN to this reader: the sentinel, and the
    // truncation it distinguishes.
    // -----------------------------------------------------------------------

    const SSE: &str = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",",
        "\"created\":1700000000,\"model\":\"demo\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        ": keepalive\n\n",
        "data: [DONE]\n\n",
    );

    /// The body every streamed call arrives with — `render_request` puts the
    /// gateway form's answer here, and `post_stream` refuses anything else.
    fn streamed() -> CreateChatCompletionRequest {
        CreateChatCompletionRequest {
            stream: Some(true),
            ..Default::default()
        }
    }

    async fn stream_from(server: &MockServer) -> Result<StreamOutcome> {
        post_stream(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            Some(&key()),
            &streamed(),
            // Unbounded, which is the default and what every test but the
            // stall one below wants: a mock answers instantly, so a limit
            // here could only ever fire spuriously.
            None,
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

    /// A stream that stops without its sentinel is TRUNCATED, and says so.
    ///
    /// The bug this pins is a silent one: without the distinction, a connection
    /// that dropped mid-answer ends in exactly the `None` a finished stream
    /// ends in, so every surface above — the bindings, and the gateway's
    /// `[DONE]` — reports a half-written reply as a complete one.
    #[tokio::test]
    async fn a_socket_that_closes_before_the_sentinel_is_an_error() {
        let server = MockServer::start().await;
        // The first chunk of SSE, and then nothing: no sentinel, no close
        // frame, just a body that stops.
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(SSE.split(": keepalive").next().unwrap().to_string()),
            )
            .mount(&server)
            .await;

        let StreamOutcome::Ok(mut stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        // Everything that DID arrive still reaches the caller first.
        assert_eq!(stream.next().await.unwrap().unwrap().id, "chatcmpl-1");
        assert!(
            matches!(
                stream.next().await,
                Some(Err(Error::StreamTruncated { provider: "demo" }))
            ),
            "a stream that stopped early must not end like one that finished"
        );
        assert!(stream.next().await.is_none(), "the error is terminal");
    }

    /// ...but a body whose last line is the sentinel, missing only the blank
    /// line that would dispatch it, FINISHED — the socket closing is what
    /// dispatches it, so the flush has to be read before truncation is
    /// declared. A body stopping mid-line is a different thing and stays
    /// truncated: half a line is what a dead connection looks like.
    #[tokio::test]
    async fn a_sentinel_without_its_trailing_blank_line_still_ends_cleanly() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(SSE.strip_suffix('\n').unwrap().to_string()),
            )
            .mount(&server)
            .await;

        let StreamOutcome::Ok(mut stream) = stream_from(&server).await.unwrap() else {
            panic!("a 200 did not stream");
        };
        assert_eq!(stream.next().await.unwrap().unwrap().id, "chatcmpl-1");
        assert!(stream.next().await.is_none(), "the sentinel did arrive");
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

    /// A stream that goes QUIET — no bytes, and no close either — is ended by
    /// `read_timeout` and nothing else.
    ///
    /// This is the failure the client's total deadline answers only by
    /// outliving: the connection is up, the provider is simply not talking.
    /// It needs a raw socket because a mock server cannot produce it —
    /// wiremock's `set_delay` holds back the whole response, which stalls the
    /// request before its headers and never reaches the read this bounds.
    ///
    /// The event sent first carries `"choices":[]`, which is a real chunk
    /// OpenAI sends (the usage chunk) and not an ending — so this also pins
    /// that emptiness is never what ends a stream.
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
            let event = concat!(
                "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",",
                "\"created\":1700000000,\"model\":\"demo\",\"choices\":[]}\n\n"
            );
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
            "demo",
            &format!("http://{addr}"),
            Some(&key()),
            &streamed(),
            Some(LIMIT),
        )
        .await
        .unwrap() else {
            panic!("a 200 did not stream");
        };

        assert_eq!(stream.next().await.unwrap().unwrap().id, "chatcmpl-1");
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            matches!(
                &error,
                Error::StreamStalled {
                    provider: "demo",
                    timeout,
                } if *timeout == LIMIT
            ),
            "a quiet socket is a stall, not a truncation or a network error: {error:?}"
        );
        assert!(stream.next().await.is_none(), "the error is terminal");
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

        for body in [None, Some(false)] {
            let refused = post_stream(
                &reqwest::Client::new(),
                "demo",
                &server.uri(),
                Some(&key()),
                &CreateChatCompletionRequest {
                    stream: body,
                    ..Default::default()
                },
                None,
            )
            .await;
            assert!(
                matches!(refused, Err(Error::InvalidInput(_))),
                "{body:?} opened a stream: {refused:?}"
            );
        }
    }

    /// ...and the mirror: a whole reply cannot carry chunks, so asking for
    /// them here is refused rather than silently unasked.
    #[tokio::test]
    async fn an_unstreamed_call_refuses_a_body_that_asks_for_a_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(SSE))
            .expect(0)
            .mount(&server)
            .await;

        let refused = post(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            Some(&key()),
            &CreateChatCompletionRequest {
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
            Some(&key()),
            &CreateChatCompletionRequest::default(),
        )
        .await;
        assert!(matches!(err, Err(Error::Decode { .. })), "{err:?}");
    }
}
