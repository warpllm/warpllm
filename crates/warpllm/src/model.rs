//! Parsing of `provider/model` strings into a provider and upstream model name.

use crate::error::{Error, Result};

/// Scaling plan (agreed, deferred until the 3rd/4th provider lands): the
/// provider roster will grow to hundreds+ of entries that are mostly
/// metadata over ~a dozen wire protocols. At that point this becomes a
/// static `phf::Map<&str, ProviderSpec>` (name → base URL, env var, auth
/// style, protocol family), and this enum shrinks to the protocol families
/// only — so the exhaustive `match` in `client.rs` stays small forever
/// while adding a provider is one line of data. Do not add per-provider
/// enum variants beyond the point where their behavior is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Provider {
    OpenAi,
}

/// Splits `"provider/model"` into the provider and the upstream model name.
///
/// Bare model names (no `/`) are rejected rather than defaulting to a
/// provider: silently assuming OpenAI becomes a footgun once many providers
/// exist. `split_once` keeps any `/` inside the model name intact.
pub(crate) fn parse_model(model: &str) -> Result<(Provider, &str)> {
    let invalid = || Error::InvalidModel {
        given: model.to_string(),
    };
    let Some((provider, name)) = model.split_once('/') else {
        return Err(invalid());
    };
    let kind = match provider {
        "openai" => Provider::OpenAi,
        _ => return Err(invalid()),
    };
    if name.is_empty() {
        return Err(invalid());
    }
    Ok((kind, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai() {
        assert_eq!(
            parse_model("openai/gpt-4o").unwrap(),
            (Provider::OpenAi, "gpt-4o")
        );
    }

    #[test]
    fn keeps_slashes_in_model_name() {
        assert_eq!(
            parse_model("openai/org/custom-model").unwrap(),
            (Provider::OpenAi, "org/custom-model")
        );
    }

    #[test]
    fn rejects_bare_model() {
        let msg = parse_model("gpt-4o").unwrap_err().to_string();
        assert!(msg.contains("not a supported provider"), "{msg}");
    }

    #[test]
    fn rejects_unknown_provider() {
        let msg = parse_model("mistral/large").unwrap_err().to_string();
        assert!(msg.contains("mistral/large"), "{msg}");
        assert!(msg.contains("not a supported provider"), "{msg}");
    }

    #[test]
    fn rejects_empty_model_name() {
        assert!(parse_model("openai/").is_err());
    }
}
