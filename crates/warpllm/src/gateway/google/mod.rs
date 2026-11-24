//! Conversions between Google's `generateContent` wire shapes
//! ([`crate::protocol::google`]) and the gateway forms, plus the vocabulary
//! glue the conversions share.
//!
//! Laid out like its [`anthropic`](super::anthropic) sibling: [`api`] holds one
//! module per API surface, named for its [`Api`](crate::Api) variant.
//!
//! EGRESS ONLY, like Anthropic: warpllm speaks this protocol to a provider and
//! answers its own callers in theirs, so there is no `ingest_request` and no
//! retained residue to prefer over the gateway form. The error-body conversion
//! arrives with the response conversion (subtask 4 of #35).

pub(crate) mod api;

use serde_json::Value;

use crate::gateway::types::{ProviderExt, Role};
use crate::protocol::UnknownFields;
use crate::types::Protocol;

/// Flattens the ext bags relevant to this render target: the protocol
/// namespace first, overlaid by the provider's own. Other providers' (and other
/// protocols') namespaces are ignored.
///
/// A near-copy of `anthropic::merged_ext`, deliberately not shared: see that
/// function for why. Both `google` the protocol and `google` the provider are
/// the same string, so the overlay is a no-op for that provider, and
/// [`Protocol::may_read`] stops the symmetric leak from a provider named for
/// another protocol.
pub(crate) fn merged_ext(ext: &ProviderExt, provider: &str) -> UnknownFields {
    let mut merged = UnknownFields::new();
    for namespace in [Protocol::Google.as_str(), provider] {
        if !Protocol::Google.may_read(namespace) {
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

/// Gateway role → Google's two-role vocabulary.
///
/// There is no `system` role (the system prompt is the top-level
/// `systemInstruction`, hoisted before this is asked) and no `tool` role (a
/// tool result is a `functionResponse` part inside a `user` turn).
pub(crate) fn role_to_wire(role: Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        Role::System | Role::User | Role::Tool => "user",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn merged_ext_ignores_other_providers_and_protocols() {
        let mut ext = ProviderExt::new();
        ext.insert("google".into(), json!({"safetySettings": []}));
        ext.insert("anthropic".into(), json!({"top_k": 40}));
        ext.insert("deepseek".into(), json!({"ds_only": 2}));

        let merged = merged_ext(&ext, "google");
        assert_eq!(merged["safetySettings"], json!([]));
        assert!(merged.get("top_k").is_none());
        assert!(merged.get("ds_only").is_none());
    }

    #[test]
    fn a_provider_named_for_another_protocol_does_not_leak_its_bag() {
        let mut ext = ProviderExt::new();
        ext.insert("anthropic".into(), json!({"top_k": 40}));
        ext.insert("google".into(), json!({"seed": 7}));
        let merged = merged_ext(&ext, "anthropic");
        assert!(merged.get("top_k").is_none());
        assert_eq!(merged["seed"], json!(7));
    }

    #[test]
    fn roles_map_onto_the_two_wire_roles() {
        assert_eq!(role_to_wire(Role::Assistant), "model");
        for role in [Role::System, Role::User, Role::Tool] {
            assert_eq!(role_to_wire(role), "user");
        }
    }
}
