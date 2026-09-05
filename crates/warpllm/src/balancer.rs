/// Smooth weighted round-robin load balancing across providers.
///
/// One [`Balancer`] per balanced group. Bounded by the number of candidates
/// — a handful, from user config — so no unbounded structures grow at runtime.
///
/// Algorithm: Nginx-style smooth weighted round-robin. Each candidate holds a
/// `current_weight` (atomic, starts at 0). On each selection:
///
/// 1. Every candidate's `current_weight` is incremented by its effective weight.
/// 2. The candidate with the highest `current_weight` is selected.
/// 3. The selected candidate's `current_weight` is decremented by the total weight.
///
/// This produces exact distribution over each cycle (length = total weight) with
/// maximum smoothness — no two identical picks are more than
/// `ceil(total / max_weight)` apart.
///
/// Thread safety: each candidate's `current_weight` is an independent
/// [`AtomicI32`]. Two threads may select the same candidate (benign — one extra
/// request to that provider), but the distribution remains correct over the cycle.
use std::sync::atomic::{AtomicI32, Ordering};

use crate::error::{Error, Result};

/// One candidate in a balanced model's rotation.
///
/// The `model_str` and its weight, and nothing else. It used to carry the
/// resolved `&'static ProviderSpec`/`ModelSpec` pair as well, both
/// `#[allow(dead_code)]`: the selected candidate's `model_str` is written back
/// onto the request and the inner client resolves it again through its own four
/// gates, so the pair was never read. Once a roster could belong to one client
/// those references stopped being `'static` at all, and threading a lifetime
/// through the balancer to carry data nobody looks at would have been the wrong
/// half of the trade.
#[derive(Debug)]
pub struct Candidate {
    pub model_str: String,
    pub weight: u32,
}

/// Smooth weighted round-robin balancer for one balanced group.
///
/// Built once at [`BalancedClient`](crate::balanced::BalancedClient)
/// construction. The candidate set is static and bounded, so no runtime
/// allocation beyond the initial construction.
///
/// # Distribution example
///
/// For candidates with weights `[3, 1]` (total = 4):
///
/// ```text
/// Step 1: A=3, B=1  → pick A → A=-1, B=1   → A
/// Step 2: A=2, B=2  → pick A → A=-2, B=2   → A
/// Step 3: A=1, B=3  → pick B → A=1,  B=-1  → B
/// Step 4: A=4, B=0  → pick A → A=0,  B=0   → A
/// ```
///
/// Cycle: A, A, B, A — exactly 75%/25%, perfectly interleaved.
#[derive(Debug)]
pub struct Balancer {
    candidates: Vec<Candidate>,
    /// Per-candidate current weight. `AtomicI32` because `current_weight`
    /// goes negative during the cycle (e.g., A=3-4=-1 after first pick).
    current: Vec<AtomicI32>,
    /// Total weight across all candidates. Stored once, used on every
    /// decrement step.
    total: i32,
}

impl Balancer {
    /// Build a balancer from a resolved candidate list.
    ///
    /// Candidates must be non-empty — validated by
    /// [`BalancedClient::new`](crate::balanced::BalancedClient::new).
    ///
    /// This is the ONE place weight arithmetic is validated. Both
    /// `BalancedClient` (Rust callers) and `JsonBalancedClient` (Python and
    /// Node, where a caller-supplied `weight` arrives as an unvalidated `u32`
    /// straight from JSON) build their candidate list and call through here,
    /// so a bindings caller cannot reach `select()`'s `i32` arithmetic with a
    /// value it was never checked against.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if any candidate's weight exceeds `i32::MAX`
    /// (the type `select()` computes in), if the weights sum to more than
    /// `i32::MAX` (`total` would overflow), or if every candidate is
    /// weight-0 (a single zero-weight candidate among positive ones is a
    /// coherent "never pick this one"; an all-zero set has no candidate
    /// left to pick, so `select()` would silently always return the first
    /// entry regardless of what the caller asked for).
    pub fn new(candidates: Vec<Candidate>) -> Result<Self> {
        let mut total: i32 = 0;
        for c in &candidates {
            let weight = i32::try_from(c.weight).map_err(|_| {
                Error::InvalidInput(format!(
                    "candidate {:?} has weight {}, which exceeds the maximum of {}",
                    c.model_str,
                    c.weight,
                    i32::MAX
                ))
            })?;
            total = total.checked_add(weight).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "candidate weights sum to more than {}; reduce them so the total fits",
                    i32::MAX
                ))
            })?;
        }
        if total == 0 {
            return Err(Error::InvalidInput(
                "all candidates have weight 0; at least one needs a positive weight".into(),
            ));
        }
        let current = candidates.iter().map(|_| AtomicI32::new(0)).collect();
        Ok(Self {
            candidates,
            current,
            total,
        })
    }

    /// Select the next candidate via smooth weighted round-robin.
    ///
    /// Lock-free: each candidate's `current_weight` is an independent atomic.
    /// Two threads may select the same candidate (benign — one extra request
    /// to that provider), but the distribution remains correct over the cycle.
    pub fn select(&self) -> &Candidate {
        // Step 1: increment every candidate's current_weight.
        for (i, c) in self.candidates.iter().enumerate() {
            self.current[i].fetch_add(c.weight as i32, Ordering::Relaxed);
        }
        // Step 2: pick the candidate with the highest current_weight.
        // `-(i as i32)` breaks ties by preferring the lower index, matching
        // the first-match semantics of the original loop.
        let best_idx = self
            .current
            .iter()
            .map(|w| w.load(Ordering::Relaxed))
            .enumerate()
            .max_by_key(|&(i, w)| (w, -(i as i32)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Step 3: decrement the winner by total weight.
        self.current[best_idx].fetch_sub(self.total, Ordering::Relaxed);
        &self.candidates[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `name` IS the `model_str` here. These cases are about which candidate
    /// comes out in which order, so the string is an opaque label and a
    /// provider-shaped one would only make every assertion longer.
    fn candidate(name: &'static str, weight: u32) -> Candidate {
        Candidate {
            model_str: name.to_string(),
            weight,
        }
    }

    #[test]
    fn single_candidate_always_selects() {
        let balancer = Balancer::new(vec![candidate("a", 1)]).unwrap();
        for _ in 0..100 {
            assert_eq!(balancer.select().model_str, "a");
        }
    }

    #[test]
    fn two_equal_weight_candidates_alternate() {
        let balancer = Balancer::new(vec![candidate("a", 1), candidate("b", 1)]).unwrap();
        let picks: Vec<&str> = (0..6)
            .map(|_| balancer.select().model_str.as_str())
            .collect();
        assert_eq!(picks, vec!["a", "b", "a", "b", "a", "b"]);
    }

    #[test]
    fn three_to_one_ratio() {
        let balancer = Balancer::new(vec![candidate("a", 3), candidate("b", 1)]).unwrap();
        let mut counts = [0u32; 2];
        for _ in 0..1000 {
            let c = balancer.select();
            if c.model_str == "a" {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }
        // Exact distribution: 750/250 over each cycle of 4.
        assert_eq!(
            counts[0], 750,
            "weight-3 candidate should be selected 750 times"
        );
        assert_eq!(
            counts[1], 250,
            "weight-1 candidate should be selected 250 times"
        );
    }

    #[test]
    fn distribution_is_exact_over_cycle() {
        // Weights [2, 1, 1], total = 4. Over 4 picks, expect 2/1/1.
        let balancer = Balancer::new(vec![
            candidate("a", 2),
            candidate("b", 1),
            candidate("c", 1),
        ])
        .unwrap();
        let mut counts = [0u32; 3];
        for _ in 0..4 {
            let c = balancer.select();
            let idx = match c.model_str.as_str() {
                "a" => 0,
                "b" => 1,
                "c" => 2,
                _ => unreachable!(),
            };
            counts[idx] += 1;
        }
        assert_eq!(counts, [2, 1, 1]);
    }

    #[test]
    fn smoothness_no_two_identical_picks_are_far_apart() {
        // Weights [5, 1], total = 6. Max gap between A picks is ceil(6/5) = 2.
        let balancer = Balancer::new(vec![candidate("a", 5), candidate("b", 1)]).unwrap();
        let picks: Vec<&str> = (0..12)
            .map(|_| balancer.select().model_str.as_str())
            .collect();
        // Find gaps between consecutive A picks.
        let a_positions: Vec<usize> = picks
            .iter()
            .enumerate()
            .filter(|(_, name)| **name == "a")
            .map(|(i, _)| i)
            .collect();
        for pair in a_positions.windows(2) {
            let gap = pair[1] - pair[0];
            assert!(
                gap <= 2,
                "gap between A picks should be at most 2, got {gap}: {picks:?}"
            );
        }
    }

    /// A single zero-weight candidate among positive ones is coherent — it
    /// is simply never picked — so construction succeeds and `select()`
    /// never returns it.
    #[test]
    fn a_single_zero_weight_candidate_is_never_selected() {
        let balancer = Balancer::new(vec![candidate("a", 1), candidate("b", 0)]).unwrap();
        for _ in 0..100 {
            assert_eq!(balancer.select().model_str, "a");
        }
    }

    /// An all-zero candidate set has no positive weight to pick by, which
    /// would otherwise make `select()` silently always return the first
    /// entry regardless of what the caller asked for.
    #[test]
    fn an_all_zero_candidate_set_is_rejected() {
        let err = Balancer::new(vec![candidate("a", 0), candidate("b", 0)]).unwrap_err();
        assert!(err.to_string().contains("weight 0"), "{err}");
    }

    /// A weight the balancer's `i32` arithmetic cannot represent is rejected
    /// at construction rather than silently inverting the distribution —
    /// `u32::MAX as i32` is `-1`, which would make this candidate lose every
    /// round instead of winning almost every one.
    #[test]
    fn a_weight_exceeding_i32_max_is_rejected() {
        let err = Balancer::new(vec![candidate("a", u32::MAX), candidate("b", 1)]).unwrap_err();
        assert!(err.to_string().contains("exceeds the maximum"), "{err}");
    }

    /// Weights that individually fit `i32` but overflow it once summed are
    /// rejected rather than panicking (`overflow-checks` on) or silently
    /// wrapping `total` negative (release, `overflow-checks` off).
    #[test]
    fn weights_summing_past_i32_max_are_rejected() {
        let err = Balancer::new(vec![
            candidate("a", 2_000_000_000),
            candidate("b", 2_000_000_000),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("sum to more than"), "{err}");
    }
}
