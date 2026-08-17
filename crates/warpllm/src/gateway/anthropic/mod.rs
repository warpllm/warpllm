//! Conversions between Anthropic's Messages wire shapes
//! ([`crate::protocol::anthropic`]) and the gateway forms, plus the vocabulary
//! glue both directions share.
//!
//! Laid out like its [`openai_compat`](super::openai_compat) sibling, and for
//! the same reasons:
//!
//! * [`api`] — one module per API surface, named for its [`Api`](crate::Api)
//!   variant.
//! * [`error`] — the error envelope's meaning, protocol-wide rather than
//!   per-surface.
//!
//! What is NOT here is a `provider_overrides` directory. `anthropic` is the
//! only provider speaking this protocol, and it matches it by definition — a
//! provider cannot diverge from a protocol it defines. #24 (Bedrock) and #25
//! (Vertex) are where that grows, and both differ in transport and auth rather
//! than in vocabulary.
//!
//! # The one asymmetry with openai_compat
//!
//! That protocol is one warpllm both SPEAKS and ANSWERS IN: callers program
//! against an OpenAI-compatible surface, so its error module has an egress half
//! that renders a warpllm failure as OpenAI would have reported it. Anthropic
//! is egress-only — warpllm speaks it to a provider and answers its own callers
//! in their protocol — so there is no counterpart here. An Anthropic-shaped
//! INGRESS would add one; that is deliberately out of scope.

pub(crate) mod api;
pub(crate) mod error;

use serde_json::Value;

use crate::gateway::types::{CacheHint, ProviderExt, Role};
use crate::protocol::UnknownFields;
use crate::protocol::anthropic::messages::types::CacheControl;
use crate::types::Protocol;

/// Flattens the ext bags relevant to this render target: the protocol
/// namespace first, overlaid by the provider's own. Other providers'
/// namespaces are ignored — a field meant for one provider never leaks into
/// another.
///
/// A near-copy of `openai_compat::merged_ext`, deliberately NOT a shared helper
/// taking a [`Protocol`]. Generalizing working code to serve a second caller is
/// a refactor of the first, and these two will diverge before they converge:
/// this protocol has one provider today and the other has five. Revisit at the
/// third.
///
/// One quirk this protocol has and that one does not: the protocol name and its
/// only provider's name are the SAME string, so both loop passes read the same
/// bag and the overlay is a no-op. That is what `ProviderExt`'s doc means by
/// provider names shadowing protocol names being reserved — here the shadow is
/// intended, since a field addressed to "anthropic" is addressed to both.
///
/// It is intended only HERE, though, which is why [`Protocol::may_read`] guards
/// the loop rather than the shadow being left to the name alone: the same bag
/// read by a renderer for another protocol is that protocol reaching into this
/// one's residue. The guard is a no-op on this protocol's own name and stops
/// the symmetric case, a provider named `openai_compat`.
pub(crate) fn merged_ext(ext: &ProviderExt, provider: &str) -> UnknownFields {
    let mut merged = UnknownFields::new();
    for namespace in [Protocol::Anthropic.as_str(), provider] {
        if !Protocol::Anthropic.may_read(namespace) {
            continue;
        }
        if let Some(Value::Object(fields)) = ext.get(namespace) {
            for (key, value) in fields {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    merged
}

/// The inverse of [`merged_ext`] for this protocol's own namespace. An empty
/// bag is omitted rather than stored as an empty object.
pub(crate) fn namespaced(fields: UnknownFields) -> ProviderExt {
    let mut ext = ProviderExt::new();
    if !fields.is_empty() {
        ext.insert(Protocol::Anthropic.as_str().into(), Value::Object(fields));
    }
    ext
}

/// Wire role string → gateway role, plus the raw spelling when it is one this
/// protocol does not define.
///
/// Anthropic has exactly TWO roles. There is no `system` — the system prompt is
/// a top-level parameter — and no `tool`, because a tool result is a block
/// inside a user turn. So the gateway's four roles do not map onto this
/// protocol's two, and the difference is handled by the request conversion
/// rather than smuggled through a role table: see `request::ingest_message` for
/// how a user turn carrying only tool results becomes [`Role::Tool`].
pub(crate) fn role_from_wire(role: String) -> (Role, Option<String>) {
    match role.as_str() {
        "user" => (Role::User, None),
        "assistant" => (Role::Assistant, None),
        _ => (Role::User, Some(role)),
    }
}

/// Gateway role → Anthropic's two-role vocabulary.
///
/// [`Role::Tool`] is `"user"`: Anthropic carries a tool result as a
/// `tool_result` BLOCK inside a user turn, which is what `render_request`'s
/// merge produces.
///
/// [`Role::System`] never reaches here. `render_request` hoists a leading
/// system message into the top-level `system` parameter and REFUSES one that
/// appears later, so by the time messages are rendered none is left — and the
/// arm is `"user"` only because that is the one non-assistant role this
/// protocol has, not because mapping a system prompt onto a user turn would be
/// acceptable. That refusal is the reason this function has nothing to decide.
pub(crate) fn role_to_wire(role: Role) -> &'static str {
    match role {
        Role::Assistant => "assistant",
        Role::System | Role::User | Role::Tool => "user",
    }
}

/// Anthropic's cache breakpoint → the gateway's.
///
/// Only `ttl` survives, because only `ttl` has a field: the `type`
/// (`"ephemeral"`) and anything else on the object are residue. That is lossy
/// in one direction and harmless in practice — a same-protocol render replays
/// the message's retained wire content rather than rebuilding from the hint, so
/// the residue is only unreachable when the request came from ANOTHER protocol,
/// which had no `type` to lose.
pub(crate) fn cache_hint(control: &CacheControl) -> CacheHint {
    CacheHint {
        ttl: control.ttl.clone(),
    }
}

/// The inverse of [`cache_hint`]. `type` is `"ephemeral"` because that is the
/// only value Anthropic documents and a breakpoint must name one.
pub(crate) fn cache_control(hint: &CacheHint) -> CacheControl {
    CacheControl {
        r#type: "ephemeral".into(),
        ttl: hint.ttl.clone(),
        unknown_fields: UnknownFields::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The isolation claim in [`merged_ext`]'s doc, at the function level: a
    /// field addressed to one provider must not reach another.
    #[test]
    fn merged_ext_ignores_other_providers() {
        let mut ext = ProviderExt::new();
        ext.insert(
            Protocol::Anthropic.as_str().into(),
            json!({"top_k": 40, "metadata": {"user_id": "u1"}}),
        );
        ext.insert("openai_compat".into(), json!({"seed": 7}));
        ext.insert("deepseek".into(), json!({"ds_only": 2}));

        let merged = merged_ext(&ext, "anthropic");
        assert_eq!(merged["top_k"], json!(40));
        assert_eq!(merged["metadata"], json!({"user_id": "u1"}));
        assert!(
            merged.get("seed").is_none(),
            "the openai_compat namespace leaked into an Anthropic render"
        );
        assert!(merged.get("ds_only").is_none());
    }

    /// The shadowing case [`merged_ext`]'s doc calls out: protocol and provider
    /// are one string here, so reading the bag twice must be a no-op rather
    /// than something that duplicates or drops.
    #[test]
    fn the_shared_protocol_and_provider_name_reads_one_bag() {
        let mut ext = ProviderExt::new();
        ext.insert("anthropic".into(), json!({"top_k": 40}));
        assert_eq!(merged_ext(&ext, "anthropic"), merged_ext(&ext, "other"));
    }

    /// The mirror of the openai_compat case: this renderer must not read that
    /// protocol's bag either, however the provider is named. Without the guard
    /// a provider literally called `openai_compat` would pull OpenAI's residue
    /// into an Anthropic body.
    #[test]
    fn a_provider_named_for_another_protocol_does_not_leak_its_bag() {
        let mut ext = ProviderExt::new();
        ext.insert(
            Protocol::OpenAiCompat.as_str().into(),
            json!({"seed": 7, "content": "a string, not blocks"}),
        );
        ext.insert(Protocol::Anthropic.as_str().into(), json!({"top_k": 40}));

        let merged = merged_ext(&ext, Protocol::OpenAiCompat.as_str());
        assert!(merged.get("seed").is_none());
        assert!(merged.get("content").is_none());
        assert_eq!(merged["top_k"], json!(40));
    }

    #[test]
    fn namespaced_omits_an_empty_bag() {
        assert!(namespaced(UnknownFields::new()).is_empty());
        assert_eq!(
            namespaced(json!({"top_k": 40}).as_object().unwrap().clone())["anthropic"],
            json!({"top_k": 40})
        );
    }

    /// The two roles this protocol names round trip with no residue. The other
    /// two gateway roles deliberately do NOT — they have no wire spelling here
    /// — which is why this asserts over a pair rather than over [`Role`].
    #[test]
    fn the_two_wire_roles_round_trip_with_no_residue() {
        for role in [Role::User, Role::Assistant] {
            assert_eq!(role_from_wire(role_to_wire(role).to_string()), (role, None));
        }
    }

    #[test]
    fn an_unknown_wire_role_keeps_its_raw_spelling() {
        let (role, raw) = role_from_wire("developer".into());
        assert_eq!(role, Role::User);
        assert_eq!(raw.as_deref(), Some("developer"));
    }

    /// A hint with no ttl still renders a breakpoint: the ttl is what is
    /// optional, the breakpoint itself is the whole point of the object.
    #[test]
    fn a_cache_hint_round_trips_through_the_wire_shape() {
        for ttl in [Some("1h".to_string()), None] {
            let hint = CacheHint { ttl: ttl.clone() };
            let control = cache_control(&hint);
            assert_eq!(control.r#type, "ephemeral");
            assert_eq!(cache_hint(&control).ttl, ttl);
        }
    }
}
