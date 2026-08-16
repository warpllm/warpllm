//! End-to-end gateway tests: a real axum server on an ephemeral port in
//! front of a wiremock "OpenAI" upstream.

use std::future::Future;
use std::sync::Arc;

use serde_json::{Value, json};
use warpllm_server::{AppState, router};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The gateway-held provider key, placed in the environment by
/// [`with_gateway_key`] — the client's only key source.
const GATEWAY_KEY: &str = "sk-gateway";

/// Runs `body` holding temp-env's lock, with `OPENAI_API_KEY` set to `key`.
///
/// EVERY test in this binary goes through this, including the ones that never
/// reach the upstream. Building a gateway READS the environment — `Client::new`
/// resolves its providers once, up front — so a test that sets nothing is still
/// a reader, and a reader running beside a writer is the data race that made
/// `set_var` unsafe in edition 2024. Only a shared lock rules it out.
///
/// These used to split: key-reading tests took the lock and the rest stayed
/// `#[tokio::test]` for parallelism, which was sound while keys resolved at
/// request time and nothing else touched the environment. It stopped being
/// sound the moment construction started reading.
///
/// A runtime per test rather than `#[tokio::test]` because `async_with_vars`
/// cannot hold the lock across an await.
///
/// `WARPLLM_SPECS` is cleared too: a contributor who exports it points every
/// gateway spawned here at a roster this suite has never seen, and the failures
/// that follow name nothing that would lead back to the variable.
fn with_env<F: Future<Output = ()>>(key: Option<&str>, body: F) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_vars([("OPENAI_API_KEY", key), ("WARPLLM_SPECS", None)], || {
        runtime.block_on(body)
    });
}

/// The gateway holding its provider key, for the tests that reach upstream.
fn with_gateway_key<F: Future<Output = ()>>(body: F) {
    with_env(Some(GATEWAY_KEY), body);
}

/// No provider key at all, for the tests that must fail before key resolution
/// (bad route, `stream: true`, unknown model, health). Proving that from an
/// environment where the key is ABSENT is a stronger claim than proving it from
/// one where the key merely went unused.
fn without_key<F: Future<Output = ()>>(body: F) {
    with_env(None, body);
}

/// Serves the gateway against the given upstream, returning its base URL.
async fn spawn_app(upstream_uri: &str) -> String {
    let client = warpllm::Client::new(warpllm::ClientConfig {
        base_url: Some(upstream_uri.to_string()),
        timeout_secs: Some(5),
        ..Default::default()
    })
    .unwrap();
    let app = router(AppState {
        client: Arc::new(client),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn request_body() -> Value {
    json!({
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "hi"}]
    })
}

fn completion_body() -> Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-5.6-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!", "refusal": null},
            "finish_reason": "stop",
            "logprobs": null
        }]
    })
}

#[test]
fn non_stream_happy_path_uses_gateway_key_and_echoes_model() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {GATEWAY_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(1)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        // No Authorization header needed: the gateway holds the provider key.
        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["model"], "openai/gpt-5.6");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello there!");

        let sent: Value =
            serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["model"], "gpt-5.6");
    });
}

#[test]
fn unprefixed_route_is_404() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
    });
}

#[test]
fn caller_bearer_is_ignored_never_forwarded() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("authorization", format!("Bearer {GATEWAY_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(1)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        // The caller sends its own bearer; the upstream must still see the
        // gateway's key (the mock 404s any other Authorization value).
        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .bearer_auth("sk-caller")
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    });
}

/// An upstream failure, reported in OpenAI's vocabulary rather than the
/// provider's. The status here already matches what OpenAI would send, so it
/// survives unchanged — what the gateway proves is that the body a caller
/// reads is the same one the in-process SDK hands back.
#[test]
fn an_upstream_failure_is_reported_in_openai_vocabulary() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .insert_header("x-request-id", "req-upstream-1")
                    .set_body_json(json!({
                        "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
                    })),
            )
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        // The upstream's own Retry-After is re-emitted as a real header:
        // standard clients and proxies back off on the header and never
        // read warpllm's JSON.
        assert_eq!(
            response.headers().get("retry-after").unwrap(),
            "30",
            "the upstream's Retry-After did not survive the gateway"
        );

        let body: Value = response.json().await.unwrap();
        // `type` and `code` are OpenAI's own spellings, identical to what a
        // caller reaching warpllm in-process would see. warpllm's taxonomy
        // rides beside them, on this surface only.
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
        assert_eq!(body["error"]["origin"], "provider");
        assert_eq!(body["error"]["warpllm_code"], "rate_limited");
        assert_eq!(body["error"]["provider"], "openai");
    });
}

/// The finding this split exists for, end to end through the HTTP gateway:
/// a quota exhaustion arrives as a 429 and reads exactly like a rate limit,
/// so a caller reading only the status backs off against a billing problem.
#[test]
fn quota_exhaustion_is_not_reported_as_a_rate_limit() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota"
                }
            })))
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["error"]["code"], "insufficient_quota",
            "reported as a rate limit, a backoff loop never resolves this"
        );
        assert_eq!(body["error"]["origin"], "provider");
        // warpllm's own name for it stays reachable for anyone debugging.
        assert_eq!(body["error"]["warpllm_code"], "quota_exceeded");
    });
}

/// The chunks an upstream sends for "Hello there!", mirroring
/// `fixtures/transcript/openai-text.sse`.
const UPSTREAM_STREAM: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",",
    "\"created\":1700000000,\"model\":\"gpt-5.6\",\"choices\":[{\"index\":0,",
    "\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"logprobs\":null,",
    "\"finish_reason\":null}],\"usage\":null,\"obfuscation\":\"KtQ3nZ8w\"}\n\n",
    ": keepalive\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",",
    "\"created\":1700000000,\"model\":\"gpt-5.6\",\"choices\":[{\"index\":0,",
    "\"delta\":{\"content\":\" there!\"},\"logprobs\":null,",
    "\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// The `data:` payloads of an SSE body, in order.
fn payloads(sse: &str) -> Vec<&str> {
    sse.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

async fn post_stream(gateway: &str, body: &Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .unwrap()
}

fn streaming_request_body() -> Value {
    let mut body = request_body();
    body["stream"] = json!(true);
    body
}

/// The gap this closes: an OpenAI SDK pointed at the gateway asks for a stream
/// and gets one, framed the way it expects.
#[test]
fn stream_true_is_served_as_sse() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {GATEWAY_KEY}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(UPSTREAM_STREAM),
            )
            .expect(1)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = post_stream(&gateway, &streaming_request_body()).await;

        assert_eq!(response.status(), 200);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );

        let body = response.text().await.unwrap();
        let payloads = payloads(&body);
        assert_eq!(payloads.len(), 3, "two chunks and the sentinel: {body}");
        assert_eq!(*payloads.last().unwrap(), "[DONE]");

        let mut text = String::new();
        for payload in &payloads[..2] {
            let chunk: Value = serde_json::from_str(payload).unwrap();
            // The caller's prefixed string, not the upstream echo.
            assert_eq!(chunk["model"], "openai/gpt-5.6");
            // Per-chunk residue no specification names still reaches the caller.
            assert_eq!(chunk["object"], "chat.completion.chunk");
            if let Some(fragment) = chunk["choices"][0]["delta"]["content"].as_str() {
                text.push_str(fragment);
            }
        }
        assert_eq!(text, "Hello there!");

        // The gateway asked the provider to stream, and stripped the prefix.
        let sent: Value = upstream.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert_eq!(sent["stream"], json!(true));
        assert_eq!(sent["model"], "gpt-5.6");
    });
}

/// ...and an EXPLICIT `"stream": false` is a whole reply, like its absence.
///
/// The other whole-reply tests all omit the key, so they only ever exercise
/// `None`. `Some(false)` is the third input the route can see, and the one a
/// loosened condition — `stream.is_some()`, `unwrap_or(true)` — would silently
/// start answering with events that a caller calling `.json()` cannot read.
#[test]
fn stream_false_is_served_as_a_whole_reply() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let mut body = request_body();
        body["stream"] = json!(false);
        let response = post_stream(&gateway, &body).await;

        assert_eq!(response.status(), 200);
        assert_ne!(
            response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
            "an explicit false must not open a stream"
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "Hello there!");
    });
}

/// A failure BEFORE the stream opens still has a status to use, so it keeps
/// its real one — and everything that makes it actionable.
#[test]
fn a_refusal_before_the_stream_keeps_its_status_and_retry_after() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_json(json!({
                        "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
                    })),
            )
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = post_stream(&gateway, &streaming_request_body()).await;

        assert_eq!(response.status(), 429);
        assert_eq!(
            response.headers().get("retry-after").unwrap(),
            "30",
            "the header a standard client backs off on"
        );
        let body: Value = response.json().await.unwrap();
        // Prefixed with the provider and status, exactly as the whole-reply
        // path renders it — a streamed refusal is not a different failure.
        assert_eq!(
            body["error"]["message"],
            "openai returned HTTP 429: Rate limit reached"
        );
        assert_eq!(body["error"]["warpllm_code"], "rate_limited");
    });
}

/// ...and a failure AFTER it opens has none left, so it travels in the body.
///
/// The upstream sends an event that will not decode. The caller must be told,
/// and must be able to tell this from a completed answer — so the error
/// arrives as an event and NO sentinel follows it.
#[test]
fn a_failure_mid_stream_arrives_as_an_error_event_with_no_sentinel() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string("data: not json\n\n"),
            )
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = post_stream(&gateway, &streaming_request_body()).await;

        // The status was committed before the body existed; it cannot change.
        assert_eq!(response.status(), 200);
        let body = response.text().await.unwrap();
        let payloads = payloads(&body);

        assert_eq!(payloads.len(), 1, "the failure is the only event: {body}");
        let error: Value = serde_json::from_str(payloads[0]).unwrap();
        assert_eq!(error["error"]["code"], "decode_error");
        assert!(
            !body.contains("[DONE]"),
            "a sentinel would claim the answer completed: {body}"
        );
    });
}

/// An upstream that DROPS mid-answer is a failure too, and the one that hides
/// best: nothing is malformed, the connection simply stops.
///
/// The chunks that arrived are forwarded — they are real — and then the
/// truncation is reported as an error event. Emitting `[DONE]` instead, which
/// is what a stream ending indistinguishably from a finished one produces,
/// would tell an OpenAI SDK the half-written answer it holds is the whole one.
#[test]
fn an_upstream_that_stops_before_its_sentinel_is_not_reported_as_complete() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(UPSTREAM_STREAM.replace("data: [DONE]\n\n", "")),
            )
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = post_stream(&gateway, &streaming_request_body()).await;

        assert_eq!(response.status(), 200);
        let body = response.text().await.unwrap();
        let payloads = payloads(&body);

        assert_eq!(payloads.len(), 3, "two chunks, then the failure: {body}");
        let error: Value = serde_json::from_str(payloads[2]).unwrap();
        assert_eq!(error["error"]["code"], "stream_truncated");
        assert!(
            !body.contains("[DONE]"),
            "a truncated answer must not be signed off as finished: {body}"
        );
    });
}

/// The roster is closed on this path too: an unregistered name never opens a
/// stream, and the refusal is an ordinary 400 rather than an empty one.
#[test]
fn streaming_an_unregistered_model_is_refused_before_upstream() {
    without_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let mut body = streaming_request_body();
        body["model"] = json!("openai/not-a-model");
        let response = post_stream(&gateway, &body).await;

        assert_eq!(response.status(), 400);
        assert!(upstream.received_requests().await.unwrap().is_empty());
    });
}

#[test]
fn invalid_model_and_invalid_json_are_400s() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;
        let client = reqwest::Client::new();

        // An unregistered name is rejected by the roster, which is checked
        // before credentials — so this stays a 400 about the model even with
        // no key in the environment, rather than becoming a 401.
        let response = client
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&json!({"model": "gpt-5.6", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["origin"], "gateway");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no registered model spec")
        );

        let response = client
            .post(format!("{gateway}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body("not json")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["origin"], "gateway");
    });
}

/// Exercises the `serve` entry point the binary and bindings share: boots
/// on a free port, answers `/health`, and exits cleanly on shutdown.
#[test]
fn serve_boots_answers_health_and_shuts_down_gracefully() {
    // `serve` builds the client itself, so this reads the environment too —
    // and boots with no providers, which is not an error.
    without_key(async {
        // Reserve a free port, then release it for serve to claim. Racy in
        // principle, harmless in practice for a test.
        let port = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = warpllm_server::config::ServerConfig {
            host: "127.0.0.1".into(),
            port,
            specs: None,
            timeout_secs: 5,
            stream_read_timeout_secs: None,
        };
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(warpllm_server::serve(config, async {
            shutdown_rx.await.ok();
        }));

        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{port}/health");
        let mut health = None;
        for _ in 0..50 {
            match client.get(&url).send().await {
                Ok(response) => {
                    health = Some(response);
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        }
        assert_eq!(health.expect("server came up").status(), 200);

        shutdown_tx.send(()).unwrap();
        server.await.unwrap().unwrap();
    });
}

#[test]
fn health_reports_ok() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .get(format!("{gateway}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], warpllm::version());
    });
}

/// A gateway serving a model only the operator's own roster knows about.
///
/// The whole `--specs` path, end to end and through a real axum server: the
/// file names a self-hosted host, the caller routes `local/…` at the gateway,
/// and the request reaches that host carrying no credential. Nothing in this
/// test is authenticated — `without_key` runs it with the environment empty —
/// so a regression that started demanding one would fail here rather than in
/// somebody's cluster.
///
/// `base_url` stays absent, unlike every other case in this file: the roster's
/// own address is the thing under test, and a global override would replace it.
#[test]
fn a_roster_file_makes_a_self_hosted_model_routable_through_the_gateway() {
    without_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "llama-3.3-70b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from the box"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&upstream)
            .await;

        let dir = tempfile::TempDir::new().unwrap();
        let roster = dir.path().join("warpllm.yaml");
        std::fs::write(
            &roster,
            format!(
                "providers:\n  local:\n    base_url: \"{}\"\n    auth: none\n    \
                 models:\n      local/llama-3.3-70b:\n        supported_apis:\n          \
                 - {{api: openai_compat_chat_completions}}\n",
                upstream.uri()
            ),
        )
        .unwrap();

        // Built the way `serve` builds one, from a `ServerConfig` — so this
        // covers the flag reaching the client, not just the field existing.
        let config = warpllm_server::config::ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            specs: Some(roster),
            timeout_secs: 5,
            stream_read_timeout_secs: None,
        };
        let app = router(AppState {
            client: Arc::new(warpllm::Client::new(config.client_config()).unwrap()),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "model": "local/llama-3.3-70b",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{:?}", response.text().await);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["model"], "local/llama-3.3-70b");

        let sent = &upstream.received_requests().await.unwrap()[0];
        assert!(
            sent.headers.get("authorization").is_none(),
            "the gateway invented a credential for a host that wants none"
        );
    });
}

/// The merge guarantee, from the gateway's side: pointing it at a roster must
/// not cost it the providers it shipped with.
#[test]
fn a_roster_file_leaves_the_gateways_built_in_providers_routable() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header(
                "authorization",
                format!("Bearer {GATEWAY_KEY}").as_str(),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&upstream)
            .await;

        let dir = tempfile::TempDir::new().unwrap();
        let roster = dir.path().join("warpllm.yaml");
        std::fs::write(
            &roster,
            "providers:\n  local:\n    base_url: \"http://127.0.0.1:1/v1\"\n    \
             auth: none\n    models:\n      local/llama-3.3-70b:\n        \
             supported_apis:\n          - {api: openai_compat_chat_completions}\n",
        )
        .unwrap();

        // The global override still points every provider at the mock, which
        // is what lets a SHIPPED model be exercised without a real key.
        let client = warpllm::Client::new(warpllm::ClientConfig {
            base_url: Some(upstream.uri()),
            specs_path: Some(roster),
            timeout_secs: Some(5),
            ..Default::default()
        })
        .unwrap();
        let app = router(AppState {
            client: Arc::new(client),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{:?}", response.text().await);
    });
}
