//! `Api::GoogleGenerateContent` ↔ the gateway forms.
//!
//! [`request`] is the conversion. The response conversion, the streamed
//! counterpart and the exchange that orders them are later subtasks of #35.

mod request;

// Nothing calls this until the transport (subtask 6 of #35) lands.
#[allow(unused_imports)]
pub(crate) use self::request::render_request;
