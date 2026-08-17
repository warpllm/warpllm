//! Client configuration.

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Holds no API key. A client reads those from the environment itself when it
/// is built, taking one variable per provider the roster names — so a client is
/// never asked up front for keys it can find, nor for keys a given request will
/// not use.
///
/// The environment is deliberately the only source for now. Carrying keys here
/// is this struct's job if an embedder that keeps them elsewhere turns up; until
/// one does, there is nothing to override.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// Overrides the provider's default base URL (proxies, tests). Absent
    /// means each provider talks to its own API.
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
    /// How long a stream may go without a single byte before warpllm gives up
    /// on it. Absent means never, which is the default.
    ///
    /// [`timeout_secs`](Self::timeout_secs) is a TOTAL deadline — it cannot
    /// tell a stream that is alive and slow from one that is wedged, so it
    /// bounds a stall only by outliving it. This bounds the GAP instead, and
    /// resets on every byte, which is the shape that fits a response whose
    /// length nobody knows in advance.
    ///
    /// Opt-in because there is no value that is right for everyone, and a
    /// wrong one fails silently in the worst direction: the gap before the
    /// FIRST chunk is a gap like any other, and a reasoning model can think
    /// for minutes before it emits a token. Set this above the slowest
    /// time-to-first-token you expect, not merely above the gap between
    /// chunks — several providers also send `:` keepalive comments during a
    /// long think, and those count as bytes.
    pub stream_read_timeout_secs: Option<u64>,
}
