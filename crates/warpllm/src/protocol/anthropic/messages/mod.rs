//! `Api::AnthropicMessages` as Anthropic speaks it: the wire shapes in
//! [`types`], the HTTP binding in `transport`.
//!
//! The conversions to and from the gateway forms live at
//! `crate::gateway::anthropic::api::messages`.

pub mod types;

// Nothing outside this module's own tests calls the transport yet: its only
// caller is `crate::gateway::anthropic::api::messages::exchange`, which lands
// with the conversions. Being unreachable is what makes this module safe to
// land ahead of them — but it also means every item here reads as dead. Drop
// this attribute when the conversions arrive; it should stop being needed on
// the same change.
#[allow(dead_code)]
pub(crate) mod transport;
