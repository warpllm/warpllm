//! Conversions between the OpenAI-compatible wire shapes
//! ([`crate::protocol::openai_compat`]) and the gateway forms, plus the
//! vocabulary glue both directions share.
//!
//! Three things, and the split is the point:
//!
//! * [`api`] — one module per API surface, named for its [`Api`](crate::Api)
//!   variant. The base implementation every provider on this protocol gets.
//! * [`error`] — the error envelope and its baseline mapping, protocol-wide
//!   rather than per-surface because every endpoint here shares one shape.
//! * [`provider_overrides`] — only where a named provider genuinely differs
//!   from the two above. Empty for most providers, and that is the healthy
//!   state.
//!
//! The shapes being converted, and the transport that carries them, belong to
//! the protocol module — this side owns only the translation.

pub(crate) mod api;
pub(crate) mod error;
mod provider_overrides;

use serde_json::Value;

use crate::gateway::types::{ProviderExt, Role};
use crate::protocol::openai_compat::chat_completions::types::UnknownFields;
use crate::types::Protocol;

/// Flattens the ext bags relevant to this render target: the protocol
/// namespace first (the target speaks it), overlaid by the provider's own
/// namespace. Other providers' namespaces are ignored — a field meant for
/// one provider never leaks into another.
///
/// And neither does another PROTOCOL's, which is a different claim and needs
/// [`Protocol::may_read`] to hold: a provider is allowed to share its name with
/// the protocol it defines, so a bag keyed `"anthropic"` may hold either
/// Anthropic passthrough or Anthropic's own retained wire fields, and this
/// renderer cannot tell them apart. Reading it would flatten a retained
/// `content` block ARRAY in beside this protocol's typed `content` string.
pub(crate) fn merged_ext(ext: &ProviderExt, provider: &str) -> UnknownFields {
    let mut merged = UnknownFields::new();
    for namespace in [Protocol::OpenAiCompat.as_str(), provider] {
        if !Protocol::OpenAiCompat.may_read(namespace) {
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

/// The inverse of [`merged_ext`] for the protocol's own namespace: untyped wire
/// fields go in under this protocol's name, so a renderer for another protocol
/// knows not to emit them. An empty bag is omitted rather than stored as an
/// empty object.
pub(crate) fn namespaced(fields: UnknownFields) -> ProviderExt {
    let mut ext = ProviderExt::new();
    if !fields.is_empty() {
        ext.insert(
            Protocol::OpenAiCompat.as_str().into(),
            Value::Object(fields),
        );
    }
    ext
}

/// Wire role string → gateway role. Unrecognized roles (e.g. OpenAI's
/// `"developer"`) map to the closest gateway role and return the raw
/// spelling so ingest can stash it for exact restoration.
pub(crate) fn role_from_wire(role: String) -> (Role, Option<String>) {
    match role.as_str() {
        "system" => (Role::System, None),
        "user" => (Role::User, None),
        "assistant" => (Role::Assistant, None),
        "tool" => (Role::Tool, None),
        _ => (Role::User, Some(role)),
    }
}

pub(crate) fn role_to_wire(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The isolation claim in [`merged_ext`]'s doc, at the function level: a
    /// field addressed to one provider must not reach another. `request.rs`
    /// covers this through `render_request`; pinning it here means a change to
    /// the merge itself fails on the merge, not three layers up.
    #[test]
    fn merged_ext_overlays_the_provider_and_ignores_other_providers() {
        let mut ext = ProviderExt::new();
        ext.insert(
            Protocol::OpenAiCompat.as_str().into(),
            json!({"shared": "from-protocol", "protocol_only": 1}),
        );
        ext.insert(
            "deepseek".into(),
            json!({"shared": "from-provider", "ds_only": 2}),
        );
        ext.insert("mistral".into(), json!({"mistral_only": true}));

        let merged = merged_ext(&ext, "deepseek");
        assert_eq!(
            merged["shared"],
            json!("from-provider"),
            "the provider namespace must overlay the protocol's"
        );
        assert_eq!(merged["protocol_only"], json!(1));
        assert_eq!(merged["ds_only"], json!(2));
        assert!(
            merged.get("mistral_only").is_none(),
            "another provider's namespace leaked into this render"
        );
    }

    /// The collision that made [`Protocol::may_read`] necessary. A provider
    /// named for a protocol addresses that protocol's bag, and this renderer
    /// cannot distinguish the two — so it must read neither.
    ///
    /// The failure this prevents is not a dropped field: Anthropic retains its
    /// whole wire `content` ARRAY under that key, `render_message` never pops
    /// `content`, and `unknown_fields` is `#[serde(flatten)]`. Reading it emits
    /// TWO `content` keys in one OpenAI message object — the typed string and
    /// the block array — which most clients resolve last-wins.
    #[test]
    fn a_provider_named_for_another_protocol_does_not_leak_its_bag() {
        let mut ext = ProviderExt::new();
        ext.insert(
            Protocol::Anthropic.as_str().into(),
            json!({"content": [{"type": "text", "text": "hi"}], "stop_sequence": null}),
        );
        ext.insert(
            Protocol::OpenAiCompat.as_str().into(),
            json!({"system_fingerprint": "fp_1"}),
        );

        let merged = merged_ext(&ext, Protocol::Anthropic.as_str());
        assert!(
            merged.get("content").is_none(),
            "Anthropic's retained content array reached an OpenAI render"
        );
        assert!(merged.get("stop_sequence").is_none());
        assert_eq!(
            merged["system_fingerprint"],
            json!("fp_1"),
            "this protocol's own bag must still be read"
        );
    }

    #[test]
    fn merged_ext_is_empty_when_no_namespace_applies() {
        let mut ext = ProviderExt::new();
        ext.insert("mistral".into(), json!({"mistral_only": true}));
        assert!(merged_ext(&ext, "deepseek").is_empty());
    }

    #[test]
    fn namespaced_files_fields_under_the_protocol_name() {
        let mut fields = UnknownFields::new();
        fields.insert("seed".into(), json!(7));

        let ext = namespaced(fields);
        assert_eq!(ext[Protocol::OpenAiCompat.as_str()], json!({"seed": 7}));
    }

    /// An empty bag is absent, not an empty object — otherwise every message
    /// without passthrough fields would carry `ext: {"openai_compat": {}}` and
    /// the gateway form would stop comparing equal to a hand-built one.
    #[test]
    fn namespaced_omits_an_empty_bag() {
        assert!(namespaced(UnknownFields::new()).is_empty());
    }

    /// [`role_from_wire`] and [`role_to_wire`] are inverses over the roles the
    /// gateway model names, with no raw spelling stashed for any of them.
    #[test]
    fn known_roles_round_trip_with_no_residue() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            assert_eq!(
                role_from_wire(role_to_wire(role).to_string()),
                (role, None),
                "{} did not survive the round trip",
                role_to_wire(role)
            );
        }
    }

    /// An unrecognized role keeps its exact wire spelling so ingest can stash
    /// it and render can restore it — which is what makes a same-protocol round
    /// trip lossless whatever role it lands on.
    ///
    /// NOTE: it lands on [`Role::User`], while `Role::System`'s own doc claims
    /// System "covers OpenAI `system` AND `developer`". The two disagree. This
    /// test pins the CODE's behaviour, not the doc's claim; nothing observable
    /// differs until a second protocol renders this message, since the raw
    /// spelling is restored verbatim for openai_compat either way.
    #[test]
    fn an_unknown_wire_role_returns_its_raw_spelling() {
        let (role, raw) = role_from_wire("developer".into());
        assert_eq!(role, Role::User);
        assert_eq!(raw.as_deref(), Some("developer"));
    }
}
