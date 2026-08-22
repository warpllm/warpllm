//! `Api::AnthropicMessages` ↔ the gateway forms.
//!
//! Split the way its openai_compat sibling is: [`request`] and [`response`] are
//! the conversions, [`stream`] is the streamed counterpart of [`response`], and
//! [`exchange`] is the one place this protocol's order — render, post, ingest —
//! is stated, for both the whole reply and the streamed one.
//!
//! [`stream`] is the one module here that does NOT promise a byte-exact round
//! trip. Its contract is reassembly-equivalence, for reasons its own docs give.

mod exchange;
mod request;
mod response;
mod stream;

// Nothing calls these yet. warpllm is CALLED in another protocol and only
// SPEAKS this one, so `ingest_request` and `render_response` wait on an
// Anthropic-shaped ingress, and `exchange` waits on the client dispatch. The
// barrel names the surface's whole vocabulary regardless — a re-export list
// tracking today's callers would churn on every wiring change, and the two
// unused halves are what the round-trip tests are written against.
#[allow(unused_imports)]
pub(crate) use self::{
    exchange::{ChatChunkStream, exchange, exchange_stream},
    request::{ingest_request, render_request},
    response::{ingest_response, render_response},
    stream::{ingest_event, render_event},
};
