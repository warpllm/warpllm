//! Routing to a server the caller runs themselves.
//!
//! The claim under test is the one issue #22 asks for: a self-hosted
//! OpenAI-compatible server — vLLM, TGI, Ollama, llama.cpp — is reachable by
//! writing a roster file and pointing a client at it, with no fork and no key.
//! Everything here is that path end to end: the file on disk, the fold over the
//! shipped roster, the credential that is deliberately absent, and the request
//! that comes out the other side.
//!
//! The upstream is wiremock rather than a real vLLM, and that is an honest
//! limit rather than a shrug: what a mock cannot prove is whether a real
//! backend's replies decode, and `tests/live_self_hosted.rs` is where that is
//! asked, of a real server, by anyone who has one. What a mock proves perfectly
//! well is everything warpllm itself does — which model string routes where,
//! which URL is built, and which headers go out. Especially the header that
//! does NOT.
//!
//! Every test is `#[test]` with a runtime built by hand rather than
//! `#[tokio::test]`, because each has to hold temp-env's lock across the whole
//! body. `Client::new` now reads TWO things from the environment — the provider
//! keys and `WARPLLM_SPECS` — and `temp_env::async_with_vars` cannot hold its
//! lock across an await, so mixing it in would let a test that sets a variable
//! run beside one asserting it is absent.

use std::collections::BTreeMap;
use std::future::Future;

use serde_json::json;
use tempfile::TempDir;
use warpllm::{
    BalancedClient, ChatCompletionRequestMessage, Client, ClientConfig,
    CreateChatCompletionRequest, Error, ProviderConfig,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A roster naming one self-hosted provider at `base_url`, taking no
/// credential — the file from the README, with the address filled in.
fn local_roster(base_url: &str) -> String {
    format!(
        "providers:\n  \
           local:\n    \
             base_url: \"{base_url}\"\n    \
             auth: none\n    \
             models:\n      \
               local/llama-3.3-70b:\n        \
                 supported_apis:\n          \
                   - {{api: openai_compat_chat_completions}}\n          \
                   - {{api: openai_compat_chat_completions_stream}}\n"
    )
}

/// Writes `yaml` into a fresh directory and hands back both. The directory has
/// to outlive the path, so it is returned rather than dropped here.
fn roster_file(yaml: &str) -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("warpllm.yaml");
    std::fs::write(&path, yaml).unwrap();
    (dir, path)
}

/// Runs `body` with the environment pinned: `vars` applied, and every variable
/// this suite cares about otherwise cleared.
///
/// `WARPLLM_SPECS` is cleared unless a case sets it. A contributor who exports
/// it would otherwise point every client built here at a roster this suite has
/// never seen, and the failures name nothing that leads back to the variable.
fn with_env<F: Future<Output = ()>>(vars: &[(&str, Option<&str>)], body: F) {
    let mut settings: Vec<(String, Option<String>)> = vec![
        ("WARPLLM_SPECS".to_string(), None),
        ("OPENAI_API_KEY".to_string(), None),
    ];
    for (var, value) in vars {
        settings.retain(|(name, _)| name != var);
        settings.push(((*var).to_string(), value.map(str::to_string)));
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_vars(settings, || runtime.block_on(body));
}

fn request(model: &str) -> CreateChatCompletionRequest {
    CreateChatCompletionRequest {
        model: model.into(),
        messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
        ..Default::default()
    }
}

/// A server that answers one chat completion, the way a local backend would.
async fn local_server() -> MockServer {
    let server = MockServer::start().await;
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
        .mount(&server)
        .await;
    server
}

fn client_with(path: &std::path::Path) -> Client {
    Client::new(ClientConfig {
        specs_path: Some(path.to_path_buf()),
        timeout_secs: Some(5),
        ..Default::default()
    })
    .unwrap()
}

// ------------------------------------------------------------- the whole path

/// The headline claim: a model warpllm has never heard of, on a host warpllm
/// has never heard of, routed by a file the caller wrote.
///
/// The Authorization assertion is the half that matters most and the half a
/// matcher cannot express — wiremock can require a header, not forbid one — so
/// the request is fished out of `received_requests` and inspected. An empty
/// `Bearer ` would satisfy every other assertion here and be rejected outright
/// by several real OpenAI-compatible servers.
#[test]
fn a_user_roster_routes_to_a_local_server_with_no_credential() {
    with_env(&[], async {
        let server = local_server().await;
        let (_dir, path) = roster_file(&local_roster(&server.uri()));

        let completion = client_with(&path)
            .chat_completions(request("local/llama-3.3-70b"))
            .await
            .expect("a roster file is all a self-hosted box should need");

        // The caller's own routing string comes back, not the upstream's echo.
        assert_eq!(completion.model, "local/llama-3.3-70b");

        let sent = &server.received_requests().await.unwrap()[0];
        assert!(
            sent.headers.get("authorization").is_none(),
            "`auth: none` must send no Authorization header at all, not an \
             empty one: {:?}",
            sent.headers.get("authorization")
        );
        // And the name that shipped is the entry's, with the provider prefix
        // stripped — the same rule every other provider is held to.
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["model"], "llama-3.3-70b");
    });
}

/// Streaming is its own surface, so serving whole replies proves nothing about
/// it — including the header, which is written by a different call.
#[test]
fn a_stream_from_a_local_server_also_carries_no_credential() {
    with_env(&[], async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(
                        "data: {\"id\":\"chatcmpl-local\",\"object\":\"chat.completion.chunk\",\
                         \"created\":1700000000,\"model\":\"llama-3.3-70b\",\"choices\":\
                         [{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n\
                         data: [DONE]\n\n",
                        "text/event-stream",
                    ),
            )
            .mount(&server)
            .await;
        let (_dir, path) = roster_file(&local_roster(&server.uri()));

        let mut stream = client_with(&path)
            .chat_completions_stream(request("local/llama-3.3-70b"))
            .await
            .expect("the roster lists the streaming surface");
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.model, "local/llama-3.3-70b");
        while stream.next().await.is_some() {}

        let sent = &server.received_requests().await.unwrap()[0];
        assert!(sent.headers.get("authorization").is_none());
    });
}

/// A stream borrows NOTHING from the client that opened it, including now the
/// roster that client loaded.
///
/// `json_client.rs` states this and both bindings depend on it — a `JsonChatStream`
/// crosses into Python and Node owning itself, with no lifetime to plumb. Until
/// now nothing held it: it was true because a provider's name was `&'static str`
/// and every other borrowed thing was already owned, so the first change that
/// gave a stream a reference into the registry would break both bindings at the
/// far end of a rebuild rather than here.
///
/// The client is dropped BEFORE a single chunk is read, which is the whole test
/// — a stream that borrowed anything would not compile past that line.
#[test]
fn a_stream_outlives_the_client_that_opened_it() {
    with_env(&[], async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(
                        "data: {\"id\":\"chatcmpl-local\",\"object\":\"chat.completion.chunk\",\
                         \"created\":1700000000,\"model\":\"llama-3.3-70b\",\"choices\":\
                         [{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n\
                         data: [DONE]\n\n",
                        "text/event-stream",
                    ),
            )
            .mount(&server)
            .await;
        let (_dir, path) = roster_file(&local_roster(&server.uri()));

        let client = client_with(&path);
        let mut stream = client
            .chat_completions_stream(request("local/llama-3.3-70b"))
            .await
            .unwrap();
        drop(client);

        let chunk = stream.next().await.unwrap().unwrap();
        // The model string is the one the caller's roster named, so the stream
        // is still carrying roster-derived text after its client is gone.
        assert_eq!(chunk.model, "local/llama-3.3-70b");
        while stream.next().await.is_some() {}
    });
}

// ------------------------------------------------------------------ the fold

/// The merge guarantee, and the reason a user file is folded rather than
/// substituted: adding a box of your own must not cost you the roster warpllm
/// ships.
#[test]
fn a_user_roster_leaves_the_shipped_providers_alone() {
    with_env(&[("OPENAI_API_KEY", Some("sk-test"))], async {
        let (_dir, path) = roster_file(&local_roster("http://127.0.0.1:1/v1"));
        let client = client_with(&path);

        let (provider, model) = client
            .fetch_model("openai/gpt-5.6")
            .expect("the shipped roster survives the fold");
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
        assert_eq!(model.model(), "gpt-5.6");

        // And the file's own entry is there beside it.
        assert_eq!(
            client.fetch_model("local/llama-3.3-70b").unwrap().0.name(),
            "local"
        );
    });
}

/// A provider entry is replaced WHOLE. Nothing in a roster is inherited, so a
/// file naming `openai:` is describing their `openai` — and the models the
/// shipped entry carried are not quietly still there under a different host.
#[test]
fn a_user_provider_replaces_a_shipped_one_whole() {
    with_env(&[], async {
        let (_dir, path) = roster_file(
            &local_roster("http://127.0.0.1:1/v1")
                .replace("  local:", "  openai:")
                .replace("local/llama", "openai/llama"),
        );
        let client = client_with(&path);

        assert_eq!(
            client
                .fetch_model("openai/llama-3.3-70b")
                .unwrap()
                .0
                .base_url(),
            "http://127.0.0.1:1/v1"
        );
        assert!(
            matches!(
                client.fetch_model("openai/gpt-5.6"),
                Err(Error::InvalidModel { .. })
            ),
            "the replaced entry's models must not survive it"
        );
    });
}

/// The registry stays closed for a host you own. A wildcard is not a special
/// case here — it never loads at all, so it cannot reach this far.
#[test]
fn a_user_roster_buys_no_wildcard() {
    with_env(&[], async {
        let (_dir, path) = roster_file(&local_roster("http://127.0.0.1:1/v1"));
        let client = client_with(&path);
        for unlisted in ["local/*", "local/llama-3.3-71b", "local"] {
            assert!(
                matches!(
                    client.fetch_model(unlisted),
                    Err(Error::InvalidModel { .. })
                ),
                "`{unlisted}` matched something"
            );
        }
    });
}

// ------------------------------------------- beside a `providers` declaration

/// A caller may declare a provider only their OWN file names.
///
/// The two features have to be checked against the same roster or they work
/// against each other: validating the declaration against the shipped list
/// would refuse somebody the provider they just wrote, which is the one thing
/// they are most certain exists.
#[test]
fn a_declaration_may_name_a_provider_from_the_callers_own_roster() {
    with_env(&[], async {
        let server = local_server().await;
        let (_dir, path) = roster_file(&local_roster(&server.uri()));
        let client = Client::new(ClientConfig {
            specs_path: Some(path),
            providers: Some(BTreeMap::from([(
                "local".to_string(),
                ProviderConfig::default(),
            )])),
            timeout_secs: Some(5),
            ..Default::default()
        })
        .expect("a declaration naming the caller's own provider is not a typo");

        assert_eq!(
            client
                .chat_completions(request("local/llama-3.3-70b"))
                .await
                .expect("the declared provider routes")
                .model,
            "local/llama-3.3-70b"
        );
        // And the narrowing still bites: declaring `local` alone withholds the
        // rest of the roster, which is what declaring one provider asks for.
        assert!(
            client
                .chat_completions(request("openai/gpt-5.6"))
                .await
                .is_err(),
            "declaring `local` left the shipped roster routable"
        );
    });
}

/// An inline key beats `auth: none`.
///
/// The roster says what the HOST wants in general; a caller who put a token in
/// front of their own box has said something more specific, and that is the
/// same precedence an inline key already had over an environment variable. The
/// alternative would make a roster line silently discard a credential the
/// caller deliberately supplied.
#[test]
fn an_inline_key_is_sent_to_a_provider_the_roster_calls_unauthenticated() {
    with_env(&[], async {
        let server = local_server().await;
        let (_dir, path) = roster_file(&local_roster(&server.uri()));
        Client::new(ClientConfig {
            specs_path: Some(path),
            providers: Some(BTreeMap::from([(
                "local".to_string(),
                ProviderConfig {
                    api_key: Some("sk-in-front-of-the-box".into()),
                },
            )])),
            timeout_secs: Some(5),
            ..Default::default()
        })
        .unwrap()
        .chat_completions(request("local/llama-3.3-70b"))
        .await
        .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        assert_eq!(
            sent.headers.get("authorization").unwrap(),
            "Bearer sk-in-front-of-the-box",
            "`auth: none` swallowed a key the caller supplied on purpose"
        );
    });
}

/// Load balancing works across a caller's OWN models.
///
/// Two boxes behind one name is the ordinary self-hosting shape, so this is
/// where the two features meet. `BalancedClient::new` used to resolve its
/// candidates through the free `fetch_model` — the SHIPPED roster — which was
/// correct while there was only one roster and became wrong the moment a client
/// could carry its own: it refused the caller's entries while
/// `client.chat_completions` served the very same string.
#[test]
fn a_balanced_client_routes_across_the_callers_own_roster() {
    with_env(&[], async {
        let server = local_server().await;
        let yaml = format!(
            "{}{}",
            local_roster(&server.uri()),
            local_roster(&server.uri())
                .replace("providers:\n  local:", "  second:")
                .replace("local/llama", "second/llama")
        );
        let (_dir, path) = roster_file(&yaml);
        let client = client_with(&path);

        let balanced = BalancedClient::new(
            &client,
            &[("local/llama-3.3-70b", 3), ("second/llama-3.3-70b", 1)],
        )
        .expect("a roster file's own models are balanceable");

        for _ in 0..4 {
            balanced
                .chat_completions(request("local/llama-3.3-70b"))
                .await
                .expect("every pick routes to the box the roster named");
        }
        // Four requests over a 3:1 cycle, all of them arriving somewhere.
        assert_eq!(server.received_requests().await.unwrap().len(), 4);
    });
}

// ------------------------------------------------------- failing at the start

/// The whole value of loading at construction: a bad roster is refused where
/// the caller is holding the path, not by a request failing hours later
/// somewhere else.
#[test]
fn a_malformed_roster_fails_at_construction() {
    with_env(&[], async {
        let (_dir, path) = roster_file("providers:\n  local:\n    base_url_typo: x\n");
        let error = Client::new(ClientConfig {
            specs_path: Some(path.clone()),
            ..Default::default()
        })
        .err()
        .expect("a roster that cannot load must not build a client");

        let message = error.to_string();
        assert!(
            message.contains(path.to_str().unwrap()),
            "the message must name the file to open: {message}"
        );
        assert!(message.contains("unknown field"), "{message}");
        // Configuration, not payload: a 500 with its own code, so an operator
        // reading a log can tell it from a TLS failure.
        let wire: serde_json::Value = serde_json::from_str(&error.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 500);
        assert_eq!(wire["error"]["code"], "invalid_roster");
    });
}

/// A roster that parses and still could not serve a request. `base_url` with a
/// trailing slash is exactly the typo a self-hosted operator makes, and it
/// loads perfectly well before producing a doubled slash on the wire.
#[test]
fn an_unusable_roster_fails_at_construction_too() {
    with_env(&[], async {
        let (_dir, path) = roster_file(&local_roster("http://localhost:8000/v1/"));
        let message = Client::new(ClientConfig {
            specs_path: Some(path),
            ..Default::default()
        })
        .err()
        .expect("a doubled slash is caught before it reaches a socket")
        .to_string();
        assert!(message.contains("doubled slash"), "{message}");
    });
}

/// The `base_url` mistakes a stranger actually makes, refused where they can
/// still see the file.
///
/// All four used to build a client and fail on the first request. The query
/// string is the one worth singling out: it does not fail at all. Appending an
/// endpoint to `http://host/v1?token=x` gives
/// `http://host/v1?token=x/chat/completions`, which is a live request to `/v1`
/// with the endpoint buried in the query — answered, wrongly, by whatever is
/// listening.
#[test]
fn a_base_url_an_endpoint_cannot_be_appended_to_fails_at_construction() {
    with_env(&[], async {
        for (base_url, expected) in [
            // What a server prints in its own startup log, pasted verbatim.
            ("localhost:8000/v1", "speaks HTTP"),
            ("not a url at all", "is not a URL"),
            ("http://127.0.0.1:1/v1?token=x", "query string"),
            ("http://127.0.0.1:1/v1#frag", "fragment"),
        ] {
            let (_dir, path) = roster_file(&local_roster(base_url));
            let message = Client::new(ClientConfig {
                specs_path: Some(path),
                ..Default::default()
            })
            .err()
            .unwrap_or_else(|| panic!("`{base_url}` built a client that cannot serve a request"))
            .to_string();
            assert!(message.contains(expected), "`{base_url}`: {message}");
        }
    });
}

#[test]
fn a_missing_roster_file_fails_at_construction() {
    with_env(&[], async {
        let message = Client::new(ClientConfig {
            specs_path: Some("./no-such-roster.yaml".into()),
            ..Default::default()
        })
        .err()
        .expect("a path that names nothing is not a roster")
        .to_string();
        assert!(message.contains("no-such-roster.yaml"), "{message}");
    });
}

// ------------------------------------------------------ where the path comes from

/// The container case: a roster mounted into an image whose entrypoint nobody
/// wants to change.
#[test]
fn warpllm_specs_names_the_roster_when_the_config_does_not() {
    let server = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { MockServer::start().await });
    let (_dir, path) = roster_file(&local_roster(&server.uri()));
    with_env(&[("WARPLLM_SPECS", Some(path.to_str().unwrap()))], async {
        let client = Client::new(ClientConfig::default()).unwrap();
        assert_eq!(
            client.fetch_model("local/llama-3.3-70b").unwrap().0.name(),
            "local"
        );
    });
}

/// What the caller passed wins over the environment, which is the only order
/// that lets a test — or a deployment — override an ambient setting.
#[test]
fn an_explicit_specs_path_beats_the_environment() {
    let (_ambient_dir, ambient) = roster_file(&local_roster("http://127.0.0.1:1/v1"));
    let (_dir, chosen) = roster_file(
        &local_roster("http://127.0.0.1:2/v1")
            .replace("  local:", "  chosen:")
            .replace("local/llama", "chosen/llama"),
    );
    with_env(
        &[("WARPLLM_SPECS", Some(ambient.to_str().unwrap()))],
        async {
            let client = client_with(&chosen);
            assert!(client.fetch_model("chosen/llama-3.3-70b").is_ok());
            assert!(
                client.fetch_model("local/llama-3.3-70b").is_err(),
                "the environment's roster was loaded despite an explicit path"
            );
        },
    );
}

/// An exported-and-blank variable is a configuration mistake, not an answer —
/// the same reading `Credentials` gives an empty API key. It must not be taken
/// for a path, which would fail to read and refuse to build a client.
#[test]
fn an_empty_warpllm_specs_is_unset() {
    with_env(&[("WARPLLM_SPECS", Some(""))], async {
        let client = Client::new(ClientConfig::default())
            .expect("an empty variable names no roster, so the shipped one stands");
        assert!(client.fetch_model("openai/gpt-5.6").is_ok());
    });
}
