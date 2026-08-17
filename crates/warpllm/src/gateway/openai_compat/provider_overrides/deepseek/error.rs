//! DeepSeek's error mapping, where it differs from `openai_compat`.
//!
//! <https://api-docs.deepseek.com/quick_start/error_codes> (checked
//! 2026-08-04).

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper};

pub(crate) struct DeepSeek;

impl ErrorMapper for DeepSeek {
    /// DeepSeek documents 402 as an exhausted account balance. Unlike a 429
    /// it is not worth waiting out — somebody has to top up — so reading it
    /// as the protocol's residual client error would send a caller to fix a
    /// payload that was fine.
    ///
    /// A status rule and not a `code` one because DeepSeek sends no code with
    /// it, which is exactly why the protocol baseline cannot classify this
    /// and DeepSeek has to say so itself.
    ///
    /// The `type` half of this lookup is unused: DeepSeek documents its
    /// failures as bare statuses and sends no family alongside them, so 402 is
    /// genuinely the only signal there is to read.
    fn from_status_and_type(&self, status: u16, _error_type: Option<&str>) -> Option<Classified> {
        (status == 402).then_some(Error::QuotaExceeded as Classified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_402_is_an_exhausted_balance_not_a_bad_request() {
        let classified = DeepSeek
            .from_status_and_type(402, None)
            .expect("402 is DeepSeek's");
        assert!(matches!(
            classified(Box::new(crate::gateway::types::ProviderError {
                provider: "deepseek",
                status: 402,
                message: "Insufficient Balance".into(),
                error_type: None,
                provider_code: None,
                retry_after: None,
                request_id: None,
                raw_body: String::new(),
            })),
            Error::QuotaExceeded(_)
        ));
    }

    /// The delta and NOTHING else. Every other status is the protocol's to
    /// answer, and restating one here would be a second copy to keep in sync.
    #[test]
    fn no_other_status_is_claimed() {
        for status in [400, 401, 403, 404, 429, 500, 503] {
            assert!(
                DeepSeek.from_status_and_type(status, None).is_none(),
                "{status}"
            );
        }
    }

    /// DeepSeek claims a status and nothing else. `from_code` is not
    /// implemented at all, so it takes the trait's `None` default — and the
    /// `type` half of `from_status_and_type` is ignored, so every family falls
    /// through to the protocol under a status DeepSeek does not claim.
    ///
    /// Asserted against the protocol's whole published vocabulary rather than
    /// a sample: an override added here later would fail this test loudly,
    /// which is the moment to ask whether the divergence is real or a copy.
    #[test]
    fn neither_code_nor_type_is_claimed() {
        for code in [
            "context_length_exceeded",
            "content_filter",
            "content_policy_violation",
            "model_not_found",
            "rate_limit_exceeded",
            "insufficient_quota",
            "billing_hard_limit_reached",
            "invalid_api_key",
        ] {
            assert!(DeepSeek.from_code(code).is_none(), "{code}");
        }
        for error_type in [
            "rate_limit_error",
            "overloaded_error",
            "authentication_error",
            "permission_error",
            "not_found_error",
            "invalid_request_error",
            "api_error",
        ] {
            // Under 400, which DeepSeek's own status rule does not claim, so
            // the family is all there is left for it to answer on.
            assert!(
                DeepSeek
                    .from_status_and_type(400, Some(error_type))
                    .is_none(),
                "{error_type}"
            );
        }
    }

    /// ...and the other half of that claim: those tiers still ANSWER for a
    /// DeepSeek response, by way of the protocol.
    ///
    /// Every case runs under a 400, which the protocol's status tier does not
    /// claim and DeepSeek's does not either — so the answer can only have come
    /// from the code or type tier, never from a status rule shadowing it.
    #[test]
    fn protocol_codes_and_types_still_classify_for_deepseek() {
        use crate::gateway::openai_compat::error::error_from_body;

        for (code, expected) in [
            ("context_length_exceeded", "context_length_exceeded"),
            ("content_filter", "content_filter"),
            ("content_policy_violation", "content_filter"),
            ("model_not_found", "model_not_found"),
            ("rate_limit_exceeded", "rate_limited"),
            ("insufficient_quota", "quota_exceeded"),
            ("billing_hard_limit_reached", "quota_exceeded"),
            ("invalid_api_key", "authentication"),
        ] {
            let body = format!(r#"{{"error":{{"message":"m","code":"{code}"}}}}"#);
            let err = error_from_body("deepseek", 400, &body, None, None);
            assert_eq!(err.code(), expected, "code {code}");
        }

        for (error_type, expected) in [
            ("rate_limit_error", "rate_limited"),
            ("overloaded_error", "overloaded"),
            ("authentication_error", "authentication"),
            ("permission_error", "permission_denied"),
            ("not_found_error", "model_not_found"),
            ("invalid_request_error", "provider_invalid_request"),
            ("api_error", "provider_server_error"),
        ] {
            let body = format!(r#"{{"error":{{"message":"m","type":"{error_type}"}}}}"#);
            let err = error_from_body("deepseek", 400, &body, None, None);
            assert_eq!(err.code(), expected, "type {error_type}");
        }
    }

    /// The tiers meeting on one body. DeepSeek's 402 is the only rule it
    /// owns, and it must still lose to a `code` the protocol reads — the
    /// upstream naming its own failure outranks a rule about status numbers,
    /// even that provider's own rule.
    #[test]
    fn the_402_rule_loses_to_a_protocol_code_and_beats_a_protocol_type() {
        use crate::gateway::openai_compat::error::error_from_body;

        let with_code = r#"{"error":{"message":"m","code":"invalid_api_key"}}"#;
        assert_eq!(
            error_from_body("deepseek", 402, with_code, None, None).code(),
            "authentication",
            "a status rule outranked the failure the provider named"
        );

        // ...but a `type` is the weakest signal, so 402 still wins there.
        let with_type = r#"{"error":{"message":"m","type":"invalid_request_error"}}"#;
        assert_eq!(
            error_from_body("deepseek", 402, with_type, None, None).code(),
            "quota_exceeded"
        );
    }

    /// DeepSeek's whole published error table, RESOLVED — the state a caller
    /// actually observes, rather than what this file alone answers.
    ///
    /// Six of the seven are `None` above and still classify correctly here,
    /// which is the inheritance working: DeepSeek writes one rule and gets
    /// the protocol's reading of the rest. If a future edit restated one of
    /// them locally this test would keep passing while
    /// `no_other_status_is_claimed` failed, which is why both exist.
    ///
    /// Bodies are bare `{}` on purpose. DeepSeek documents these as status
    /// codes and sends no `type` or `code` alongside them, so the status is
    /// genuinely the only signal — the hardest case, not the easiest.
    ///
    /// <https://api-docs.deepseek.com/quick_start/error_codes>
    #[test]
    fn the_published_error_table_resolves_end_to_end() {
        use crate::gateway::openai_compat::error::error_from_body;

        for (status, expected) in [
            (400, "provider_invalid_request"), // Invalid Format
            (401, "authentication"),           // Authentication Fails
            (402, "quota_exceeded"),           // Insufficient Balance — the delta
            (422, "provider_invalid_request"), // Invalid Parameters
            (429, "rate_limited"),             // Rate Limit Reached
            (500, "provider_server_error"),    // Server Error
            (503, "overloaded"),               // Server Overloaded
        ] {
            let err = error_from_body("deepseek", status, "{}", None, None);
            assert_eq!(err.code(), expected, "{status}");
        }
    }

    /// The same table under `openai`, which has no overrides — identical
    /// everywhere except 402. Without this, a bug that applied DeepSeek's
    /// rule to every provider would pass every test above.
    #[test]
    fn only_the_402_differs_from_a_provider_with_no_overrides() {
        use crate::gateway::openai_compat::error::error_from_body;

        for status in [400, 401, 422, 429, 500, 503] {
            assert_eq!(
                error_from_body("deepseek", status, "{}", None, None).code(),
                error_from_body("openai", status, "{}", None, None).code(),
                "{status} diverged, but DeepSeek only overrides 402"
            );
        }
        assert_ne!(
            error_from_body("deepseek", 402, "{}", None, None).code(),
            error_from_body("openai", 402, "{}", None, None).code(),
        );
    }
}
