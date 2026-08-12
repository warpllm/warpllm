//! The four conversions for `Api::ChatCompletions` in the OpenAI-compatible
//! protocol. Ingest is lossless; render out may be lossy, and says so where it
//! is.

mod exchange;
mod request;
mod response;
mod stream;

pub(crate) use exchange::{ChatChunkStream, exchange, exchange_stream};
pub(crate) use request::{ingest_request, render_request};
pub(crate) use response::{ingest_response, render_response};
pub(crate) use stream::{ingest_chunk, render_chunk};
