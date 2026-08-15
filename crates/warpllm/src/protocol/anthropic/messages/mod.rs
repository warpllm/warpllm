//! `Api::AnthropicMessages` as Anthropic speaks it: the wire shapes in
//! [`types`], the HTTP binding in `transport`.
//!
//! The conversions to and from the gateway forms live at
//! `crate::gateway::anthropic::api::messages`.

pub mod types;

pub(crate) mod transport;
