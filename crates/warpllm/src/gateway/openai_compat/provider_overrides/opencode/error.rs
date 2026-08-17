//! OpenCode Zen's error mapping, where it differs from `openai_compat`.
//!
//! Zen wraps an OpenAI-shaped envelope around a vocabulary of its own: the
//! body is `{"type":"error","error":{"type":…,"message":…}}` with **no
//! `code`** anywhere, and the `type` is `PascalCase`-suffixed-`Error` rather
//! than any spelling the baseline knows. Envelopes below are live responses,
//! with the workspace id redacted (checked 2026-08-17).

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper};

pub(crate) struct OpenCode;

impl ErrorMapper for OpenCode {
    /// `CreditsError` is an unpaid account, not a bad key — and Zen reports it
    /// at **HTTP 401**:
    ///
    /// ```json
    /// {"type":"error","error":{"type":"CreditsError","message":"No payment method. Add a payment method here: https://opencode.ai/workspace/wrk_…/billing"}}
    /// ```
    ///
    /// That status is the whole difficulty. The baseline reads 401 as
    /// [`Error::Authentication`] and is RIGHT to — Zen answers a genuinely bad
    /// key with the same status and its own family for it:
    ///
    /// ```json
    /// {"type":"error","error":{"type":"AuthError","message":"Invalid API key."}}
    /// ```
    ///
    /// So a status rule here would be wrong in the other direction, telling
    /// somebody with an expired key to go add a payment method. The family is
    /// what separates them, and it is read ALONE rather than as a `(401,
    /// "CreditsError")` pair: `CreditsError` names one failure and one remedy
    /// wherever it appears, so pinning it to a status would only mean Zen
    /// moving it to a 402 tomorrow — the status this failure ought to have —
    /// silently restored the misclassification this rule exists to fix.
    ///
    /// Nothing else Zen sends is claimed. `AuthError` at 401 and
    /// `FreeUsageLimitError` at 429 are already what the baseline says they
    /// are, and restating them here would be a second copy to keep in sync;
    /// `the_live_error_vocabulary_resolves_end_to_end` pins both against the
    /// real classifier instead.
    fn from_status_and_type(&self, _status: u16, error_type: Option<&str>) -> Option<Classified> {
        (error_type == Some("CreditsError")).then_some(Error::QuotaExceeded as Classified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::openai_compat::error::error_from_body;

    fn zen(error_type: &str, message: &str) -> String {
        serde_json::json!({
            "type": "error",
            "error": {"type": error_type, "message": message},
        })
        .to_string()
    }

    fn coded(status: u16, error_type: &str) -> &'static str {
        error_from_body("opencode", status, &zen(error_type, "m"), None, None).code()
    }

    /// Zen's whole live vocabulary, RESOLVED — what a caller actually
    /// observes, rather than what this file alone answers.
    ///
    /// Two of the three are `None` above and still classify correctly, which
    /// is the inheritance working. They are here because they are the same
    /// "typed vocabulary the baseline cannot see" shape as the one that broke,
    /// and a future change to `classify`'s ordering would otherwise be free to
    /// move them without anything failing.
    #[test]
    fn the_live_error_vocabulary_resolves_end_to_end() {
        // The delta: a 401 that is not an authentication failure.
        assert_eq!(coded(401, "CreditsError"), "quota_exceeded");
        // ...and a 401 that is, which the baseline already reads correctly.
        assert_eq!(coded(401, "AuthError"), "authentication");
        // The per-model free-tier limit. A real 429, and worth waiting out —
        // unlike the credit exhaustion above, which never is.
        assert_eq!(coded(429, "FreeUsageLimitError"), "rate_limited");
    }

    /// THE bug (#66), stated as the two envelopes that made it hard: one
    /// status, two failures, opposite remedies. Neither signal decides alone,
    /// so a rule written on either one would misclassify the other.
    #[test]
    fn one_401_separates_into_billing_and_credentials() {
        assert_ne!(
            coded(401, "CreditsError"),
            coded(401, "AuthError"),
            "an unpaid account and a bad key classified the same, and the \
             caller is sent after the wrong fix for one of them"
        );
    }

    /// The delta and NOTHING else. Every other spelling is the protocol's to
    /// answer, including the two pinned above.
    #[test]
    fn nothing_but_the_credits_family_is_claimed() {
        for error_type in ["AuthError", "FreeUsageLimitError", "ModelError"] {
            assert!(
                OpenCode
                    .from_status_and_type(401, Some(error_type))
                    .is_none(),
                "{error_type}"
            );
        }
        for status in [400, 401, 403, 404, 429, 500, 503] {
            assert!(
                OpenCode.from_status_and_type(status, None).is_none(),
                "{status}"
            );
        }
        for code in [
            "insufficient_quota",
            "invalid_api_key",
            "rate_limit_exceeded",
        ] {
            assert!(OpenCode.from_code(code).is_none(), "{code}");
        }
    }

    /// A `code` still outranks this — the tier split `classify` keeps. Zen
    /// sends none today, but a proxy in front of it that adds one is naming
    /// the failure outright, which beats any reading of the envelope around
    /// it.
    #[test]
    fn a_code_still_outranks_the_credits_family() {
        let body = r#"{"error":{"type":"CreditsError","message":"m","code":"invalid_api_key"}}"#;
        assert_eq!(
            error_from_body("opencode", 401, body, None, None).code(),
            "authentication"
        );
    }

    /// The rule is Zen's alone. Without this, a bug that applied it to every
    /// provider would pass every test above.
    #[test]
    fn no_other_provider_reads_a_credits_family() {
        let body = zen("CreditsError", "no payment method");
        assert_eq!(
            error_from_body("openai", 401, &body, None, None).code(),
            "authentication"
        );
        assert_eq!(
            error_from_body("deepseek", 401, &body, None, None).code(),
            "authentication"
        );
    }

    /// Classifying must never consume what Zen actually said — the variant is
    /// what it MEANS, and the family and message are the evidence for it.
    /// `CreditsError`'s message carries the billing URL, which is the one
    /// thing that fixes the failure.
    #[test]
    fn the_billing_message_survives_the_classification() {
        let body = zen(
            "CreditsError",
            "No payment method. Add a payment method here: \
             https://opencode.ai/workspace/wrk_redacted/billing",
        );
        let error = error_from_body("opencode", 401, &body, None, None);
        assert!(matches!(error, Error::QuotaExceeded(_)));
        let detail = error.provider_error().expect("a provider failure");
        assert_eq!(detail.error_type.as_deref(), Some("CreditsError"));
        assert_eq!(detail.provider_code, None);
        assert!(detail.message.contains("/billing"), "{}", detail.message);
    }
}
