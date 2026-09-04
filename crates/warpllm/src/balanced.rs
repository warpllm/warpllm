//! Load-balanced client: distributes requests across provider/model pairs.
//!
//! [`BalancedClient`] wraps a [`Client`] and adds smooth weighted
//! round-robin selection via [`Balancer`](crate::balancer::Balancer). The
//! candidate set is fixed at construction; each call picks one candidate and
//! delegates to the inner client's normal 4-gate validation.

use crate::balancer::Balancer;
use crate::client::{ChatCompletionStream, Client};
use crate::error::{Error, Result};
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use std::fmt;

/// A client that load-balances across multiple provider/model pairs.
///
/// Built from a [`Client`] reference and a list of `(model_str, weight)` pairs.
/// Each incoming request is routed to the next candidate chosen by smooth
/// weighted round-robin, then handed to the inner client's normal validation
/// and execution path.
///
/// The [`Balancer`] is stateful (per-candidate `current_weight`), so
/// `BalancedClient` is not `Sync` in the general sense — but the balancer's
/// atomics make it safe to share across threads anyway.
///
/// # Example
///
/// ```no_run
/// use warpllm::{BalancedClient, Client, ClientConfig};
///
/// let client = Client::new(ClientConfig::default()).unwrap();
/// let balanced = BalancedClient::new(&client, &[
///     ("openai/gpt-5.6", 3),
///     ("deepseek/deepseek-v4-pro", 1),
/// ]).unwrap();
/// // Use balanced.chat_completions(request) instead of client.chat_completions(request)
/// ```
pub struct BalancedClient<'a> {
    client: &'a Client,
    balancer: Balancer,
}

impl fmt::Debug for BalancedClient<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BalancedClient")
            .field("balancer", &self.balancer)
            .finish()
    }
}

impl<'a> BalancedClient<'a> {
    /// Creates a new balanced client.
    ///
    /// # Arguments
    ///
    /// * `client` — The underlying client used for every request.
    /// * `candidates` — Non-empty list of `(model_str, weight)` pairs. Each
    ///   `model_str` must exist in the registry. Weight determines the relative
    ///   proportion of requests routed to that candidate.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if `candidates` is empty.
    /// - [`Error::InvalidModel`] if any `model_str` is not in the roster.
    pub fn new(client: &'a Client, candidates: &[(&str, u32)]) -> Result<Self> {
        if candidates.is_empty() {
            return Err(Error::InvalidInput(
                "balanced client requires at least one candidate".into(),
            ));
        }
        let mut resolved = Vec::with_capacity(candidates.len());
        for &(model_str, weight) in candidates {
            // THIS client's roster, not the shipped one. A caller balancing
            // across models from a roster file of their own is the ordinary
            // case for self-hosting — two boxes behind one name — and asking
            // the free `fetch_model` would refuse them their own entries while
            // `client.chat_completions` served the very same string.
            client.fetch_model(model_str)?;
            resolved.push(crate::balancer::Candidate {
                model_str: model_str.to_string(),
                weight,
            });
        }
        Ok(Self {
            client,
            balancer: Balancer::new(resolved),
        })
    }

    /// Selects the next candidate and returns a new request with the
    /// `model` field rewritten to match.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if the request carries a `models` failover
    /// list: the balancer's weighted selection IS the candidate choice, and
    /// silently dropping the caller's redundancy would leave them believing
    /// they have failover when they have none.
    fn prepare(&self, request: CreateChatCompletionRequest) -> Result<CreateChatCompletionRequest> {
        if request.models.is_some() {
            return Err(Error::InvalidInput(
                "models is not supported with BalancedClient; \
                 the balancer selects the candidate"
                    .into(),
            ));
        }
        let mut request = request;
        request.model.clone_from(&self.balancer.select().model_str);
        Ok(request)
    }

    /// Performs a non-streaming chat completion via the next balanced candidate.
    ///
    /// The request's `model` field is overwritten with the selected candidate's
    /// `model_str` before the inner client processes it — the caller's model
    /// name is the *group* name, and each candidate is a concrete provider/model
    /// within that group.
    pub async fn chat_completions(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        let request = self.prepare(request)?;
        self.client.chat_completions(request).await
    }

    /// Performs a streaming chat completion via the next balanced candidate.
    ///
    /// Same model-rewriting as [`Self::chat_completions`].
    pub async fn chat_completions_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let request = self.prepare(request)?;
        self.client.chat_completions_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balancer::Candidate;
    use crate::{Client, ClientConfig};

    #[test]
    fn empty_candidates_rejected() {
        let client = Client::new(ClientConfig::default()).unwrap();
        let err = BalancedClient::new(&client, &[]).unwrap_err();
        assert!(err.to_string().contains("at least one candidate"));
    }

    #[test]
    fn unknown_model_rejected() {
        let client = Client::new(ClientConfig::default()).unwrap();
        let err = BalancedClient::new(&client, &[("nope/nope", 1)]).unwrap_err();
        assert!(err.to_string().contains("no registered model"));
    }

    #[test]
    fn balancer_distribution_from_public_interface() {
        // Directly test the balancer that BalancedClient wraps.
        let balancer = Balancer::new(vec![
            Candidate {
                model_str: "a/test".into(),
                weight: 3,
            },
            Candidate {
                model_str: "b/test".into(),
                weight: 1,
            },
        ]);
        let mut counts = [0u32; 2];
        for _ in 0..1000 {
            let c = balancer.select();
            if c.model_str == "a/test" {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }
        assert_eq!(counts[0], 750);
        assert_eq!(counts[1], 250);
    }
}
