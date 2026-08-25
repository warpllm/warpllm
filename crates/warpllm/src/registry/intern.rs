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
//! "Distinct strings ever seen" is only a bound if the names repeat, and that
//! is an assumption about the caller rather than a property of this code. A
//! service loading a roster per tenant, with a provider named after each, sees
//! a new string every time and grows without limit — so [`CAPACITY`] makes the
//! bound real and refuses past it, counting provider names and credential
//! variables together because they share this one table. It is set far above any legitimate roster;
//! reaching it means names are being generated, not written, and that is worth
//! failing loudly rather than leaking quietly until the process dies of it.
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

/// The most distinct roster identities one process will ever leak.
///
/// Counts provider names AND the environment-variable names beside them —
/// everything this module hands out — because they share one table and it is
/// the table that has to be bounded. So the ceiling is not 4096 providers; a
/// roster whose providers each name their own variable reaches it at half
/// that, which is still three orders of magnitude above the shipped roster and
/// far above any file somebody writes by hand.
///
/// What it stops is the one shape that grows without limit: rosters whose
/// names vary per load.
const CAPACITY: usize = 4096;

/// Every identity handed out so far, so the same text is never leaked twice.
///
/// One table for provider names and credential variables together. They are
/// not distinguished because nothing downstream distinguishes them: both are
/// `&'static str` for the life of the process, and a bound that counted only
/// one of them would not bound the leak.
static NAMES: LazyLock<Mutex<HashSet<&'static str>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// `name` as a string that lives as long as the process, reusing the one
/// already leaked for that text if there is one.
///
/// Called once per provider per roster load, never on the request path.
///
/// # Errors
///
/// The message, when [`CAPACITY`] distinct names have already been leaked and
/// this is a new one. A roster already interned keeps loading — reaching the
/// cap does not break a process that stays on the rosters it has.
pub(super) fn intern(name: &str) -> Result<&'static str, String> {
    // Nothing inside the guard can panic, so the lock cannot be poisoned.
    let mut names = NAMES
        .lock()
        .expect("nothing under this lock panics, so it cannot be poisoned");
    intern_into(&mut names, name, CAPACITY)
}

/// The whole of [`intern`] except which set it fills, so the cap can be tested
/// against a small one.
///
/// It cannot be tested against the real set: this table is process-wide and the
/// unit tests share a process, so a case that filled it would leave every
/// roster loaded afterwards — in every other test — failing to intern.
fn intern_into(
    names: &mut HashSet<&'static str>,
    name: &str,
    capacity: usize,
) -> Result<&'static str, String> {
    if let Some(existing) = names.get(name) {
        return Ok(existing);
    }
    if names.len() >= capacity {
        return Err(format!(
            "`{name}`: this process has already seen {capacity} distinct roster \
             names — providers and their environment variables together — and \
             each one lives as long as the process. Rosters loaded over a \
             process's life are meant to name the same providers; if yours are \
             generated per tenant or per request, name the providers and \
             variables from a fixed set and vary the models under them instead"
        ));
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(leaked);
    Ok(leaked)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: the same text twice is the same allocation, so a
    /// process loading one roster repeatedly leaks it once.
    #[test]
    fn the_same_name_twice_is_the_same_allocation() {
        let first = intern("demo-interned").unwrap();
        let second = intern(&String::from("demo-interned")).unwrap();
        assert_eq!(first, "demo-interned");
        assert!(std::ptr::eq(first, second));
    }

    /// And the dedup is by text, not by luck — two names stay two.
    #[test]
    fn different_names_stay_different() {
        let one = intern("demo-interned-one").unwrap();
        let two = intern("demo-interned-two").unwrap();
        assert!(!std::ptr::eq(one, two));
        assert_eq!((one, two), ("demo-interned-one", "demo-interned-two"));
    }

    /// A NEW name past the cap is refused, so "bounded by distinct names ever
    /// seen" is a property of this code rather than a hope about the caller.
    #[test]
    fn a_new_name_past_the_cap_is_refused() {
        let mut names = HashSet::new();
        for i in 0..2 {
            intern_into(&mut names, &format!("capped-{i}"), 2).unwrap();
        }
        let error = intern_into(&mut names, "capped-2", 2).unwrap_err();
        assert!(error.contains("capped-2"), "{error}");
        assert!(error.contains("2 distinct roster names"), "{error}");
    }

    /// Reaching the cap does not break a process that stays on the rosters it
    /// already has: a name already interned still comes back.
    #[test]
    fn a_name_already_interned_survives_the_cap() {
        let mut names = HashSet::new();
        let first = intern_into(&mut names, "capped-again", 1).unwrap();
        let second = intern_into(&mut names, "capped-again", 1)
            .expect("an already-interned name is not a new one");
        assert!(std::ptr::eq(first, second));
        assert!(intern_into(&mut names, "capped-other", 1).is_err());
    }
}
