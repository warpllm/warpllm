//! `Api::AnthropicMessages` ↔ the gateway forms.
//!
//! Split the way its openai_compat sibling is: [`request`] and [`response`] are
//! the conversions, [`exchange`] is the one place this protocol's order —
//! render, post, ingest — is stated.
//!
//! The streaming half — the named-event mapping and the chunk stream carrying
//! it — lands separately. It is the harder piece and earns its own review
//! rather than being buried in this one.

mod exchange;
mod request;
mod response;

// Nothing calls these yet. warpllm is CALLED in another protocol and only
// SPEAKS this one, so `ingest_request` and `render_response` wait on an
// Anthropic-shaped ingress, and `exchange` waits on the client dispatch. The
// barrel names the surface's whole vocabulary regardless — a re-export list
// tracking today's callers would churn on every wiring change, and the two
// unused halves are what the round-trip tests are written against.
#[allow(unused_imports)]
pub(crate) use self::{
    exchange::exchange,
    request::{ingest_request, render_request},
    response::{ingest_response, render_response},
};
