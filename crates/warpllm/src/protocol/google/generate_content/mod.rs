//! `Api::GoogleGenerateContent` as Google speaks it: the wire shapes in
//! [`types`]. Its serde name, `google_generate_content`, matches this module
//! path (see [`crate::protocol`]). The HTTP binding follows in a later subtask
//! of #35, and the conversions to and from the gateway forms will live at
//! `crate::gateway::google::api::generate_content`.

pub mod types;
