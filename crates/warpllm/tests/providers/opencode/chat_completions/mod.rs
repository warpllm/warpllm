use crate::openai_common::{client_for, request, with_opencode_key};
use serde_json::json;
use warpllm::Error;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Zen's error envelope, as the live API sends it: an OpenAI-shaped wrapper
/// around a vocabulary of Zen's own, and no `code` anywhere.
fn zen_error(error_type: &str, message: &str) -> serde_json::Value {
    json!({"type": "error", "error": {"type": error_type, "message": message}})
}

/// Issue #66, at the level it was reported: an account with no payment method
/// is a billing failure, not a bad key.
///
/// The unit tests beside the mapper prove the vocabulary; this proves the
/// whole path a caller actually takes reaches it — routing under the
/// `opencode` name, the override table, and the classifier's ordering, none of
/// which the mapper's own tests exercise together.
///
/// Both envelopes arrive at **HTTP 401**, which is the entire difficulty: the
/// status is identical and only the family separates a bill from a
/// credential. Asserted in one test for that reason — they are one claim.
#[test]
fn one_401_separates_a_bill_from_a_bad_key() {
    with_opencode_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(zen_error(
                "CreditsError",
                "No payment method. Add a payment method here: \
                 https://opencode.ai/workspace/wrk_redacted/billing",
            )))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("opencode/kimi-k3"))
            .await
            .unwrap_err();

        assert!(
            matches!(err, Error::QuotaExceeded(_)),
            "an unpaid account was reported as {err:?}, sending the caller \
             after an API key that is fine"
        );
        let upstream = err.provider_error().expect("a provider failure");
        assert_eq!(upstream.provider, "opencode");
        assert_eq!(upstream.status, 401);
        // The remedy travels with the failure — it is the only thing that
        // fixes this one, and Zen puts it in the message.
        assert!(
            upstream.message.contains("/billing"),
            "{}",
            upstream.message
        );
    });

    with_opencode_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_json(zen_error("AuthError", "Invalid API key.")),
            )
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("opencode/kimi-k3"))
            .await
            .unwrap_err();

        // The other half of the claim. A rule reading the 401 alone would get
        // one of these two wrong whichever way it was written.
        assert!(matches!(err, Error::Authentication(_)), "{err:?}");
    });
}

/// Zen's per-model free-tier limit, which is a real rate limit and worth
/// waiting out — unlike the credit exhaustion above, which never is.
///
/// Already correct through the protocol baseline's reading of 429, and pinned
/// here because it is the same "typed vocabulary the baseline cannot see"
/// shape: a later change to how families and statuses are ranked must not move
/// it silently.
#[test]
fn a_free_tier_limit_is_a_rate_limit() {
    with_opencode_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(zen_error(
                "FreeUsageLimitError",
                "Free usage limit reached for this model.",
            )))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("opencode/kimi-k3"))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::RateLimited(_)), "{err:?}");
        assert_eq!(
            err.provider_error().expect("a provider failure").error_type,
            Some("FreeUsageLimitError".to_string())
        );
    });
}
