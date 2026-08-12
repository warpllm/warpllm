//! Shared primitives for the gateway types.

pub(crate) mod error;
pub(crate) mod message;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod stream;

/// `pub`, not `pub(crate)` like its siblings, so `lib.rs` can re-export these
/// three. It leaks nothing on its own — `crate::gateway` is a private module,
/// so the only way out is that re-export, the same arrangement `registry`
/// uses for its specs.
pub use error::*;
pub(crate) use message::*;
pub(crate) use request::*;
pub(crate) use response::*;
pub(crate) use stream::*;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::Protocol;

/// Namespaced passthrough bags, keyed by protocol name (`"openai_compat"`)
/// or provider name (`"deepseek"`). Renderers emit the protocol namespace
/// only when the target speaks that protocol, then overlay the target
/// provider's namespace — a field meant for one provider never leaks into
/// another. Provider names shadowing protocol names are reserved.
pub(crate) type ProviderExt = BTreeMap<String, Value>;

/// Exact JSON text, never a parsed tree at rest.
///
/// Tool arguments stay raw because key order is load-bearing: some
/// providers' prompt caches hash the request bytes, and a parse/reserialize
/// round trip can also destroy the not-quite-valid JSON models sometimes
/// emit. Parse on demand; re-serialize only from the raw form.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct RawJson(String);

impl RawJson {
    pub(crate) fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    /// For ingest from providers that send objects: serialize ONCE at the
    /// boundary (preserve_order keeps the provider's key order).
    #[allow(dead_code)] // staged: used by the first object-arguments protocol
    pub(crate) fn from_value(value: &Value) -> Self {
        Self(value.to_string())
    }

    #[allow(dead_code)] // staged: used once typed tools render
    pub(crate) fn parse(&self) -> Result<Value, serde_json::Error> {
        if self.0.trim().is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        serde_json::from_str(&self.0)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where a gateway value came from. Retained so adapters can one day
/// take a same-protocol passthrough fast path, and for diffing what was
/// received against what was sent. Not part of the canonical JSON form.
#[derive(Debug, Clone)]
#[allow(dead_code)] // staged: read once the passthrough fast path lands
pub(crate) struct IngestSource {
    /// The wire format this was ingested from. [`Protocol`] rather than an
    /// enum of its own: the name that keys the `ext` bags and the name of the
    /// format are one string, so they are one type. The roster no longer
    /// records a protocol at all — a surface names its own — so this is a tag
    /// the ingesting module states about itself.
    pub protocol: Protocol,
    /// The inbound body as ingested — re-serialized from the typed wire
    /// form, NOT the original bytes: unknown-field order survives
    /// (preserve_order), but whitespace, numeric spelling, and typed-field
    /// order are canonicalized. Byte-exact capture needs the gateway
    /// boundary's raw bytes and lands with the passthrough fast path.
    pub body: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_json_round_trips_and_parses() {
        let raw = RawJson::new(r#"{"z":1,"a":2}"#);
        assert_eq!(raw.as_str(), r#"{"z":1,"a":2}"#);
        assert_eq!(raw.parse().unwrap()["z"], 1);
        // from_value keeps the source order (preserve_order).
        let value = serde_json::json!({"z": 1, "a": 2});
        assert_eq!(RawJson::from_value(&value).as_str(), r#"{"z":1,"a":2}"#);
    }

    #[test]
    fn raw_json_empty_parses_to_empty_object() {
        assert_eq!(
            RawJson::new("").parse().unwrap(),
            Value::Object(Default::default())
        );
    }

    #[test]
    fn raw_json_invalid_errors_on_parse_only() {
        let raw = RawJson::new(r#"{"almost": json"#);
        assert_eq!(raw.as_str(), r#"{"almost": json"#);
        assert!(raw.parse().is_err());
    }
}
