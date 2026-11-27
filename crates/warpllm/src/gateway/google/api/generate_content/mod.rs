//! `Api::GoogleGenerateContent` ↔ the gateway forms.
//!
//! [`request`] and [`response`] are the two conversions. The streamed
//! counterpart and the exchange that orders them are later subtasks of #35.

mod request;
mod response;

// Nothing calls these until the transport (subtask 6 of #35) lands.
#[allow(unused_imports)]
pub(crate) use self::{
    request::render_request,
    response::{ingest_response, synthetic_response_id},
};
