//! Claude reached over its own protocol, end to end.
//!
//! Every other provider suite here proves ROUTING: the same openai_compat
//! endpoint serves it, so a test only has to show the prefix was stripped and
//! the key went out. This one proves TRANSLATION. The caller writes a
//! chat-completions request and reads a chat-completions reply, and in between
//! warpllm speaks a wire format the caller never sees — so what these tests
//! watch is the body that went out and the body that came back, not just the
//! fact that something did.
//!
//! Requests are built by deserializing JSON rather than by assembling typed
//! structs. That is the shape a caller's request actually arrives in through
//! both bindings and the server, and it keeps a tool-call fixture readable.

use crate::openai_common::{
    ANTHROPIC_KEY, anthropic_message_body, client_for, request, with_anthropic_key,
};
use serde_json::{Value, json};
use warpllm::{CreateChatCompletionRequest, Error};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A caller's request as JSON, which is how one really arrives.
fn from_json(body: Value) -> CreateChatCompletionRequest {
    serde_json::from_value(body).expect("the fixture is a valid chat-completions request")
}

/// A mock serving one Anthropic reply, and the body warpllm sent to get it.
async fn exchange(request: CreateChatCompletionRequest, reply: Value) -> (Value, Value) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", ANTHROPIC_KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply))
        .expect(1)
        .mount(&server)
        .await;

    let completion = client_for(&server)
        .chat_completions(request)
        .await
        .expect("a 200 is a completion");

    let sent = serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    // Serialized rather than inspected field by field: what a caller receives is
    // the JSON, and several of the claims below are about keys that must be
    // ABSENT — which a struct-level assertion cannot see.
    (sent, serde_json::to_value(completion).unwrap())
}

/// The whole point of #23, in one test. An OpenAI-shaped request reaches
/// `api.anthropic.com` as an Anthropic-shaped body, and its reply comes back
/// OpenAI-shaped.
#[test]
fn a_chat_completions_request_reaches_claude_and_returns_chat_completions() {
    with_anthropic_key(async {
        let (sent, got) =
            exchange(request("anthropic/claude-opus-5"), anthropic_message_body()).await;

        // OUT: Anthropic's shape. `messages` without the system turn, and no
        // `stream` on a whole-reply request.
        assert_eq!(
            sent["model"], "claude-opus-5",
            "the prefix reached upstream"
        );
        assert_eq!(sent["messages"][0]["role"], "user");
        assert!(sent["stream"].is_null());

        // BACK: chat completions' shape, and the caller's own model string
        // rather than the upstream echo.
        assert_eq!(got["model"], "anthropic/claude-opus-5");
        assert_eq!(got["object"], "chat.completion");
        assert_eq!(got["choices"][0]["message"]["content"], "Hello there!");
        assert_eq!(got["choices"][0]["message"]["role"], "assistant");
        assert_eq!(got["choices"][0]["finish_reason"], "stop");

        // Anthropic reports no creation time and warpllm invents none.
        assert_eq!(got["created"], 0);

        // Usage in OpenAI's names, with the cache counts folded into the input
        // the way the gateway form defines them.
        assert_eq!(got["usage"]["prompt_tokens"], 14);
        assert_eq!(got["usage"]["completion_tokens"], 12);
        assert_eq!(got["usage"]["total_tokens"], 26);
    });
}

/// The namespace collision, closed. `anthropic` is both a PROTOCOL and a
/// PROVIDER, so a bag keyed `"anthropic"` holds either Anthropic's retained
/// wire fields or provider passthrough — and until dispatch existed there was
/// no live path on which the openai_compat renderer could be handed the first
/// kind. `Protocol::may_read` is what refuses it.
///
/// Asserted on the SERIALIZED reply, and that is load-bearing: serde emits
/// duplicate keys without complaint, so a renderer that flattened Anthropic's
/// retained `content` block array in beside its own typed `content` string
/// would produce a body with two `content` keys and a struct-level check would
/// see nothing wrong.
#[test]
fn an_anthropic_reply_leaks_no_residue_to_an_openai_caller() {
    with_anthropic_key(async {
        let (_, got) = exchange(request("anthropic/claude-opus-5"), anthropic_message_body()).await;

        let text = serde_json::to_string(&got).unwrap();
        assert_eq!(
            text.matches("\"content\"").count(),
            1,
            "two `content` keys in one reply: {text}"
        );
        for leaked in ["\"type\"", "\"stop_sequence\"", "\"input_tokens\""] {
            assert!(
                !text.contains(leaked),
                "Anthropic's {leaked} reached an OpenAI caller: {text}"
            );
        }
        // And no `ext` bag is emitted under this protocol's name. The KEY
        // rather than the substring: the reply legitimately says
        // `"model": "anthropic/claude-opus-5"`, because echoing the caller's
        // own prefixed string is the one place that word belongs.
        assert!(
            !text.contains("\"anthropic\":"),
            "a namespaced bag reached the caller: {text}"
        );
    });
}

/// Anthropic REQUIRES a `max_tokens` and chat completions does not, so the
/// roster's ceiling is the fallback — and the caller's own value outranks it.
///
/// Both directions in one test: a version that always sent the roster's figure
/// passes the first assertion and silently overrides every caller who named
/// one.
#[test]
fn the_roster_supplies_the_max_tokens_anthropic_requires() {
    with_anthropic_key(async {
        let (sent, _) =
            exchange(request("anthropic/claude-opus-5"), anthropic_message_body()).await;
        // claude-opus-5's documented output ceiling, from `specs.yaml`.
        assert_eq!(sent["max_tokens"], 128_000);

        let mut asked = request("anthropic/claude-opus-5");
        asked.max_tokens = Some(Some(64));
        let (sent, _) = exchange(asked, anthropic_message_body()).await;
        assert_eq!(
            sent["max_tokens"], 64,
            "the caller's ceiling was overridden"
        );
    });
}

/// A `developer` message is a SYSTEM instruction, and Anthropic's system
/// instruction is a top-level field rather than a turn.
///
/// This is the class of bug a second protocol exists to expose. `developer`
/// used to normalize to `Role::User` with its raw spelling in the residue,
/// which was invisible while only one protocol existed — the spelling was
/// restored verbatim on the way out. Anthropic may not read that bag, so all
/// it ever saw was a user turn, and the instruction arrived as an ordinary
/// message with none of the force the caller asked for.
#[test]
fn a_developer_instruction_reaches_claude_as_its_system_prompt() {
    with_anthropic_key(async {
        let (sent, _) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [
                    {"role": "developer", "content": "Answer in French."},
                    {"role": "user", "content": "hi"}
                ]
            })),
            anthropic_message_body(),
        )
        .await;

        assert_eq!(sent["system"], "Answer in French.");
        // And it is NOT also a message, which is the half that would make the
        // conversation two consecutive user turns.
        let roles: Vec<&str> = sent["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["user"]);
    });
}

/// `max_completion_tokens` caps a Claude reply, and the roster's ceiling is
/// only a fallback.
///
/// OpenAI's newer families REJECT `max_tokens` and require this spelling, so a
/// caller moving from `openai/gpt-5-nano` to a Claude model carries it across
/// unchanged. warpllm does not model it, so it rode `ext["openai_compat"]` —
/// which the Anthropic renderer may not read. A request asking for 16 tokens
/// therefore went out asking for 128,000.
#[test]
fn the_newer_spelling_of_the_cap_is_honoured() {
    with_anthropic_key(async {
        let (sent, _) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 16
            })),
            anthropic_message_body(),
        )
        .await;
        assert_eq!(
            sent["max_tokens"], 16,
            "the caller's cap was replaced by the roster's ceiling"
        );

        // And it outranks the older spelling when a caller sends both, which is
        // OpenAI's own deprecation order.
        let (sent, _) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "max_completion_tokens": 16
            })),
            anthropic_message_body(),
        )
        .await;
        assert_eq!(sent["max_tokens"], 16);
    });
}

/// Anthropic answers an overload with **529**, which is its own status and not
/// one `classify`'s residual would place. Without the table entry it would
/// arrive as `Error::Unknown` when it is a retryable server condition.
#[test]
fn a_529_is_reported_as_an_overload() {
    with_anthropic_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(529).set_body_json(json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"}
            })))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("anthropic/claude-opus-5"))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Overloaded(_)), "{err:?}");
        // And it reaches a non-Rust caller as a 503, not as an unattributed
        // failure.
        let wire: Value = serde_json::from_str(&err.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 503);
    });
}

/// A control the caller's protocol can express and the routed one cannot is
/// REFUSED, not dropped.
///
/// warpllm's answer everywhere else is passthrough — forward what the caller
/// wrote, let the provider judge it — and a translated route quietly loses the
/// second half of that. `n` rides `ext["openai_compat"]`, which the Anthropic
/// renderer is forbidden to read, so the provider never receives it and so
/// cannot reject it. Dropped, `n: 2` comes back as one choice, which is
/// indistinguishable from a model that ignored the request.
///
/// The mock expects ZERO requests: the refusal has to land before the network,
/// or the caller is billed for an answer to a question they did not ask.
#[test]
fn a_control_anthropic_cannot_express_is_refused_not_dropped() {
    with_anthropic_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_message_body()))
            .expect(0)
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2
            })))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        assert!(format!("{err}").contains("`n`"), "{err}");
    });
}

/// The rest of the class, each one a control that changes the ANSWER rather
/// than the request's shape. Left in `ext["openai_compat"]` every one of these
/// reaches Claude as if it were never written, and the caller reads a
/// materially different completion as a normal reply.
#[test]
fn every_untranslatable_control_is_refused_before_the_network() {
    with_anthropic_key(async {
        for (field, value) in [
            ("frequency_penalty", json!(2)),
            ("presence_penalty", json!(-1.5)),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(5)),
            ("logit_bias", json!({"1234": -100})),
            ("seed", json!(42)),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_message_body()))
                .expect(0)
                .mount(&server)
                .await;

            let err = client_for(&server)
                .chat_completions(from_json(json!({
                    "model": "anthropic/claude-opus-5",
                    "messages": [{"role": "user", "content": "hi"}],
                    field: value
                })))
                .await
                .unwrap_err();

            assert!(matches!(err, Error::InvalidInput(_)), "{field}: {err:?}");
            assert!(format!("{err}").contains(field), "{field}: {err}");
        }
    });
}

/// The value an SDK fills in when the caller asked for nothing is not a
/// refusal. Every control above has one, and rejecting on the KEY rather than
/// on the value would turn away the ordinary requests that carry them —
/// which is most requests.
#[test]
fn the_default_value_of_an_untranslatable_control_still_passes() {
    with_anthropic_key(async {
        let (sent, _) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0.0,
                "logprobs": false,
                "logit_bias": {}
            })),
            anthropic_message_body(),
        )
        .await;

        // And none of them rode along into a body with no field to hold them.
        for field in ["n", "frequency_penalty", "logprobs", "logit_bias"] {
            assert!(sent.get(field).is_none(), "{field} reached Claude: {sent}");
        }
    });
}

/// Only what CHANGES the answer is refused. One choice is what Anthropic
/// returns anyway, so a caller who spelled the default out is not turned away —
/// a refusal here would reject requests every OpenAI SDK can emit by default.
#[test]
fn asking_for_the_one_choice_anthropic_returns_is_not_a_refusal() {
    with_anthropic_key(async {
        let (_, got) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 1
            })),
            anthropic_message_body(),
        )
        .await;

        assert_eq!(got["choices"].as_array().unwrap().len(), 1, "{got}");
    });
}

/// A caller who forbids parallel tool calls is obeyed across the seam.
///
/// The two protocols spell it oppositely — `parallel_tool_calls: false` here,
/// `disable_parallel_tool_use: true` there — and it hangs off `tool_choice`
/// on Anthropic's side, so this also pins that a choice gets synthesized to
/// carry it. Dropped, the model is free to open several calls at once for a
/// caller who explicitly ruled that out.
#[test]
fn forbidding_parallel_tool_calls_reaches_claude() {
    with_anthropic_key(async {
        let (sent, _) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "parallel_tool_calls": false
            })),
            anthropic_message_body(),
        )
        .await;

        assert_eq!(
            sent["tool_choice"]["disable_parallel_tool_use"],
            json!(true),
            "{sent}"
        );
        // And the caller's own spelling does not ride along into a body that
        // has no such field.
        assert!(sent.get("parallel_tool_calls").is_none(), "{sent}");
    });
}

/// The test the neutral tool path was built for, and the one Responses will
/// copy. An OpenAI-shaped conversation with tools, an assistant `tool_calls`
/// turn, and TWO consecutive `role: "tool"` results renders to a valid
/// Anthropic body — and the reply's `tool_use` comes back as `tool_calls`.
///
/// The merge is the part with nowhere else to be tested end to end: chat
/// completions sends one message per result, Anthropic wants them as blocks
/// inside ONE user turn.
#[test]
fn a_tool_conversation_crosses_both_ways() {
    with_anthropic_key(async {
        let (sent, got) = exchange(
            from_json(json!({
                "model": "anthropic/claude-opus-5",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "weather in both?"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}},
                        {"id": "call_b", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Lima\"}"}}
                    ]},
                    {"role": "tool", "tool_call_id": "call_a", "content": "-3C"},
                    {"role": "tool", "tool_call_id": "call_b", "content": "24C"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "current weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }],
                "tool_choice": "required"
            })),
            json!({
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_9", "name": "get_weather",
                     "input": {"city": "Oslo"}}
                ],
                "stop_reason": "tool_use", "stop_sequence": null,
                "usage": {"input_tokens": 5, "output_tokens": 7}
            }),
        )
        .await;

        // The system turn is HOISTED to the top level, not sent as a message.
        assert_eq!(sent["system"], "Be brief.");
        // Tools in Anthropic's spelling: `input_schema`, no function wrapper.
        assert_eq!(sent["tools"][0]["name"], "get_weather");
        assert_eq!(sent["tools"][0]["input_schema"]["type"], "object");
        // "required" is `any` here.
        assert_eq!(sent["tool_choice"]["type"], "any");

        // THE merge: user, assistant, and ONE user turn holding both results.
        let roles: Vec<&str> = sent["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
        let results = &sent["messages"][2]["content"];
        assert_eq!(results.as_array().unwrap().len(), 2);
        assert_eq!(results[0]["type"], "tool_result");
        assert_eq!(results[0]["tool_use_id"], "call_a");
        assert_eq!(results[1]["tool_use_id"], "call_b");
        // Arguments are TEXT in the IR and an OBJECT here.
        assert_eq!(sent["messages"][1]["content"][0]["input"]["city"], "Oslo");

        // BACK: a tool call in this protocol's vocabulary, with `function` and
        // not Anthropic's `tool_use`.
        let call = &got["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["id"], "toolu_9");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Oslo\"}");
        assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
    });
}

/// Every chunk of a streamed reply, rendered for the caller.
async fn stream_of(request: CreateChatCompletionRequest, transcript: &str) -> Vec<Value> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(transcript),
        )
        .mount(&server)
        .await;

    let mut stream = client_for(&server)
        .chat_completions_stream(request)
        .await
        .expect("a 200 opens a stream");
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(serde_json::to_value(chunk.expect("a well-formed transcript")).unwrap());
    }
    chunks
}

/// A transcript that opens with TEXT and then calls a tool, which is the
/// ordinary shape and the one that exposes the index divergence.
const TEXT_THEN_TOOL: &str = concat!(
    r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","#,
    r#""role":"assistant","model":"claude-opus-5","content":[],"#,
    r#""stop_reason":null,"stop_sequence":null,"#,
    r#""usage":{"input_tokens":4,"output_tokens":0}}}"#,
    "\n\n",
    r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    "\n\n",
    r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"one sec"}}"#,
    "\n\n",
    r#"data: {"type":"content_block_stop","index":0}"#,
    "\n\n",
    r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","#,
    r#""id":"toolu_1","name":"get_weather","input":{}}}"#,
    "\n\n",
    r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","#,
    r#""partial_json":"{\"city\":"}}"#,
    "\n\n",
    r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","#,
    r#""partial_json":"\"Oslo\"}"}}"#,
    "\n\n",
    r#"data: {"type":"content_block_stop","index":1}"#,
    "\n\n",
    r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
    "\n\n",
    r#"data: {"type":"message_stop"}"#,
    "\n\n",
);

/// THE streamed blocker, closed. Anthropic numbers every content block, so a
/// reply that opens with text puts its first tool call at index 1 — and
/// `tool_calls[].index` is an index INTO the array an OpenAI client assembles.
/// openai-python's own streaming accumulator indexes `tool_calls` by that
/// value, so a 1 with no 0 before it is an out-of-range failure rather than a
/// cosmetic mismatch.
///
/// End to end rather than against the renderer, because the map lives on the
/// STREAM: a per-chunk renderer cannot hold it, and nothing but an open stream
/// can prove the fragments of one call agree.
#[test]
fn a_streamed_tool_call_is_numbered_for_the_caller_not_for_anthropic() {
    with_anthropic_key(async {
        let chunks = stream_of(request("anthropic/claude-opus-5"), TEXT_THEN_TOOL).await;

        let indices: Vec<i64> = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"][0]["index"].as_i64())
            .collect();
        assert!(
            !indices.is_empty(),
            "no tool-call fragment reached the caller"
        );
        assert!(
            indices.iter().all(|&index| index == 0),
            "Anthropic's block index reached an OpenAI client as an array slot: {indices:?}"
        );

        // The opener still names the call, and the fragments still carry the
        // argument text — the remap must not have cost either.
        let text: String = chunks
            .iter()
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(text, "{\"city\":\"Oslo\"}");
        assert!(
            chunks
                .iter()
                .any(|chunk| { chunk["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1" }),
            "the opener's id was dropped"
        );
    });
}

/// A stream's totals are the caller's to ask for. Anthropic reports them on
/// every `message_delta` whether or not anyone did, and chat completions
/// reports them only under `stream_options.include_usage` — on a TRAILING chunk
/// that carries no choices.
///
/// Both sides in one test, because the risk is a gate that is wired to nothing:
/// a version ignoring the request passes whichever half it was written for.
#[test]
fn a_streams_totals_arrive_only_when_the_caller_asked() {
    with_anthropic_key(async {
        let unasked = stream_of(request("anthropic/claude-opus-5"), TEXT_THEN_TOOL).await;
        assert!(
            unasked.iter().all(|chunk| chunk["usage"].is_null()),
            "totals nobody asked for reached the caller"
        );

        let mut asked = request("anthropic/claude-opus-5");
        asked.stream_options = Some(Some(warpllm::ChatCompletionStreamOptions {
            include_usage: Some(true),
            unknown_fields: Default::default(),
        }));
        let chunks = stream_of(asked, TEXT_THEN_TOOL).await;

        let last = chunks.last().expect("a stream with chunks");
        assert_eq!(
            last["choices"].as_array().map(Vec::len),
            Some(0),
            "the totals rode a chunk that also carried content: {last}"
        );
        assert_eq!(last["usage"]["completion_tokens"], 9);
        assert_eq!(last["usage"]["prompt_tokens"], 4);
        // Exactly one chunk carries them.
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| !chunk["usage"].is_null())
                .count(),
            1
        );
    });
}

/// A model that serves Anthropic's surface and a model that serves chat
/// completions go out over DIFFERENT protocols from the same entrypoint, which
/// is what egress dispatch means.
///
/// The path is the tell and it is the whole assertion: `/messages` against
/// `/chat/completions`. A dispatch that ignored the routed model's surface
/// would send both to whichever arm it hard-coded, and every other test here
/// would still pass.
#[test]
fn the_routed_models_surface_picks_the_protocol_not_the_entrypoint() {
    with_anthropic_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_message_body()))
            .expect(1)
            .mount(&server)
            .await;

        client_for(&server)
            .chat_completions(request("anthropic/claude-sonnet-5"))
            .await
            .unwrap();

        // And the OpenAI-compatible model on the same client, whose key is
        // absent here — so it must be REFUSED before a request goes out rather
        // than sent to `/messages`.
        let err = client_for(&server)
            .chat_completions(request("openai/gpt-5.6"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::MissingApiKey { .. }), "{err:?}");
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "a second request went out"
        );
    });
}
