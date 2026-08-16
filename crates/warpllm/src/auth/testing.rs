//! What an [`Authenticator`] actually puts on a request, for tests elsewhere.
//!
//! Goes through a real [`reqwest::Request`] rather than reading the variant's
//! fields: what these tests are about is the wire, and the fields are private
//! anyway. Shared rather than copied because `credentials` asserts the same
//! thing this module does — that the right secret reached the right provider
//! under the right header — and a second copy would be a second thing to keep
//! in step with the schemes as they grow.

use super::Authenticator;

/// The value `authenticator` sets for `name`, or `None` if it sets that header
/// at all.
pub(crate) async fn applied(authenticator: &Authenticator, name: &str) -> Option<String> {
    let request = reqwest::Client::new()
        .post("https://example.invalid/v1/chat/completions")
        .body("{}")
        .build()
        .expect("a well-formed request");
    authenticator
        .authenticate(request)
        .await
        .expect("a header credential cannot fail on a valid secret")
        .headers()
        .get(name)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}
