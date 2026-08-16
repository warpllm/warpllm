//! Process-lifetime storage for the two roster strings that outlive the roster
//! they came from.
//!
//! A client's roster is its own — an `Arc<Registry>` built when the client was,
//! dropped when it is. Almost everything in it can live there: `base_url`, the
//! model keys, the upstream names, the limits. Two strings cannot, and it is
//! worth being precise about which and why, because "leak it" is otherwise the
//! kind of shortcut that spreads.
//!
//! A provider's NAME reaches the public error surface. [`ProviderError`] holds
//! `provider: &'static str`, and five [`Error`] variants carry the same field;
//! between them that lifetime threads through `http`, the gateway's error
//! mapping, both transports, and [`ChatCompletionStream`], which is documented
//! as borrowing nothing from the client that opened it. Its env-var name is the
//! same story one variant along, in `Error::MissingApiKey`. Making those
//! `Arc<str>` would reference-count a dozen short strings that are never freed
//! anyway, at the cost of some twenty signatures and a breaking change to a
//! public field.
//!
//! So they are interned instead, and the justification is not merely that it is
//! cheaper. A provider's name is not per-client data. Two clients loading the
//! same roster mean the SAME provider, and a third loading a different roster
//! that names it again means it a third time — the name is a fact about the
//! world rather than about the file it was read from. That is exactly what a
//! process-lifetime string is for.
//!
//! The leak is bounded by DISTINCT strings ever seen, not by clients: a process
//! that builds a million clients from one roster interns the same handful.
//!
//! # The ceiling, stated
//!
//! This is deliberately unbounded storage, so here is exactly what bounds it in
//! practice and exactly what would stop doing so.
//!
//! What lands here is a provider name and an env-var name, from a file an
//! OPERATOR chose — `specs.yaml` plus at most one `specs_path`, read once when a
//! client is built. No request can reach this function, and nothing re-reads a
//! roster after construction, so the total is a property of the deployment's
//! configuration rather than of its traffic or its uptime. For every shape
//! warpllm has today that is a handful of strings and a few hundred bytes,
//! whether the process lives for a second or a year.
//!
//! Three changes would each break that, and each one is a reason to stop
//! interning rather than to raise a limit:
//!
//! - **A roster per tenant**, with tenant-scoped provider names. Then the total
//!   grows with tenants, which is a population, not a configuration.
//! - **Hot reload.** Names then churn with time rather than being fixed at
//!   startup, and a renamed provider never gives its old name back.
//! - **Rosters from untrusted input** — a hosted service taking an uploaded
//!   file. Then the growth is attacker-chosen, which makes this a
//!   memory-exhaustion vector rather than a footnote.
//!
//! The fix in any of those cases is the same and is known: `Arc<str>` for
//! provider identity. It costs roughly twenty signatures and a breaking change
//! to [`ProviderError`]'s public `provider` field, which is why it is not paid
//! in advance — but whoever builds one of the three above must pay it in the
//! same change, not after. `a_stream_outlives_the_client_that_opened_it` in
//! `tests/self_hosted.rs` pins the constraint that refactor has to respect:
//! `Arc<str>` satisfies it, a borrow out of the registry does not.
//!
//! One honest imperfection meanwhile: interning happens in `build_provider`,
//! which runs before [`lint::usable`](super::lint), so a roster that is rejected
//! has already interned its names. Deferring it would mean carrying an
//! un-interned `ProviderSpec` through validation — a parallel representation of
//! the whole type — to save a few dozen bytes on a startup that is about to fail
//! anyway. Not worth the second shape.
//!
//! [`ProviderError`]: crate::ProviderError
//! [`Error`]: crate::Error
//! [`ChatCompletionStream`]: crate::ChatCompletionStream

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

/// Every name handed out so far, so the same text is never leaked twice.
static NAMES: LazyLock<Mutex<HashSet<&'static str>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// `name` as a string that lives as long as the process, reusing the one
/// already leaked for that text if there is one.
///
/// Called once per provider per roster load, never on the request path.
pub(super) fn intern(name: &str) -> &'static str {
    // Nothing inside the guard can panic, so the lock cannot be poisoned.
    let mut names = NAMES
        .lock()
        .expect("nothing under this lock panics, so it cannot be poisoned");
    if let Some(existing) = names.get(name) {
        return existing;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: the same text twice is the same allocation, so a
    /// process loading one roster repeatedly leaks it once.
    #[test]
    fn the_same_name_twice_is_the_same_allocation() {
        let first = intern("demo-interned");
        let second = intern(&String::from("demo-interned"));
        assert_eq!(first, "demo-interned");
        assert!(std::ptr::eq(first, second));
    }

    /// And the dedup is by text, not by luck — two names stay two.
    #[test]
    fn different_names_stay_different() {
        let one = intern("demo-interned-one");
        let two = intern("demo-interned-two");
        assert!(!std::ptr::eq(one, two));
        assert_eq!((one, two), ("demo-interned-one", "demo-interned-two"));
    }
}
