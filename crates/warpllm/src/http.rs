//! Shared HTTP helpers used by every provider implementation.

use std::time::Duration;

use crate::error::{Error, Result};

/// Server-sent event framing, with no socket in it: bytes in, payloads out.
///
/// Shared by every streaming protocol rather than living beside one of them,
/// because the framing is the same wherever it is used and the fiddly part is
/// worth testing once — a read boundary falls wherever TCP puts it, which is
/// routinely mid-line and can be mid-character. Keeping it free of I/O means
/// every edge below is a plain function call in a test.
///
/// It reads the subset providers actually send:
///
/// - `data:` lines accumulate into the current event, joined by newlines when
///   a provider splits one across several (the specification allows it; none
///   of them do it today);
/// - a blank line dispatches the accumulated event;
/// - `:` comment lines — the usual keepalive — and every other SSE field
///   (`event:`, `id:`, `retry:`) are ignored;
/// - the [`sentinel`](Self::new), where a protocol has one, ends the stream
///   rather than becoming an event.
///
/// Ignoring `event:` is what lets Anthropic reuse this unchanged: it names
/// every event on that line AND repeats the name in the payload's own `type`,
/// so the framing can stay a pure `data:` reader.
#[derive(Debug)]
pub(crate) struct SseFrames {
    /// Read but not yet consumed as complete lines.
    buffer: Vec<u8>,
    /// The `data:` payload of the event being accumulated.
    event: String,
    /// The payload that ends this protocol's stream, for protocols that end
    /// one with a payload. `None` where a protocol instead ends its stream
    /// with a normal event the caller must decode to recognize — Anthropic's
    /// `message_stop` — which is a decision only that protocol's reader can
    /// make, since the framing never parses what it dispatches.
    sentinel: Option<&'static str>,
    /// The sentinel arrived. Never set when there is no sentinel to arrive.
    pub(crate) done: bool,
}

impl SseFrames {
    pub(crate) fn new(sentinel: Option<&'static str>) -> Self {
        Self {
            buffer: Vec::new(),
            event: String::new(),
            sentinel,
            done: false,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next dispatched payload, or `None` when the bytes so far do not
    /// contain a complete event — which is the signal to go read more.
    ///
    /// Lines are decoded as UTF-8 only once complete, so a multi-byte
    /// character split across two reads is rejoined rather than mangled.
    // `std::result::Result` spelled out: the crate's own `Result` alias fixes
    // the error type, and this one is a UTF-8 failure the caller labels.
    pub(crate) fn next_payload(
        &mut self,
    ) -> std::result::Result<Option<String>, std::str::Utf8Error> {
        while !self.done {
            let Some(line) = self.take_line() else {
                return Ok(None);
            };
            let line = std::str::from_utf8(&line)?;
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                if let Some(payload) = self.dispatch() {
                    return Ok(Some(payload));
                }
            } else if let Some(payload) = line.strip_prefix("data:") {
                if !self.event.is_empty() {
                    self.event.push('\n');
                }
                // One optional leading space belongs to the framing.
                self.event
                    .push_str(payload.strip_prefix(' ').unwrap_or(payload));
            }
        }
        Ok(None)
    }

    /// The event accumulated so far, for a socket that closed without sending
    /// its final blank line — providers do end streams that way.
    pub(crate) fn flush(&mut self) -> Option<String> {
        self.dispatch()
    }

    /// The next complete line, `\n` removed, or `None` when the buffer holds
    /// only a partial one.
    fn take_line(&mut self) -> Option<Vec<u8>> {
        let end = self.buffer.iter().position(|byte| *byte == b'\n')?;
        let mut line: Vec<u8> = self.buffer.drain(..=end).collect();
        line.pop();
        Some(line)
    }

    fn dispatch(&mut self) -> Option<String> {
        let payload = std::mem::take(&mut self.event);
        let payload = payload.trim();
        if payload.is_empty() {
            return None;
        }
        if Some(payload) == self.sentinel {
            self.done = true;
            return None;
        }
        Some(payload.to_string())
    }
}

/// Header spellings for the upstream's own request identifier, in the order
/// they are tried.
///
/// Two entries, both de-facto standards (OpenAI sends the first, Anthropic
/// the second), and the list stays short ON PURPOSE. A long table of
/// per-provider header names is the staleness liability the classifier
/// already refuses to take on — a spelling nobody here knows costs a `None`
/// on a diagnostic field, never a wrong answer.
const REQUEST_ID_HEADERS: [&str; 2] = ["x-request-id", "request-id"];

/// Maps a transport-level reqwest error to [`Error::Network`].
pub(crate) fn network_error(provider: &'static str, source: reqwest::Error) -> Error {
    Error::Network { provider, source }
}

/// A response read to the end: its status, its body, and the header
/// evidence that would otherwise be dropped on the floor.
///
/// The headers are read HERE rather than at the conversion layer because
/// this is the last place they exist — a body and a status alone cannot
/// answer "how long should I wait" or "what do I quote to the provider's
/// support desk".
pub(crate) struct ResponseParts {
    pub status: u16,
    pub body: String,
    /// `Retry-After`, when the provider sent one in delta-seconds form.
    pub retry_after: Option<Duration>,
    /// The upstream's own request identifier, for correlating with a
    /// provider's logs.
    pub request_id: Option<String>,
}

/// Reads the response body and the headers worth keeping, mapping read
/// failures to [`Error::Network`].
pub(crate) async fn read_response(
    provider: &'static str,
    response: reqwest::Response,
) -> Result<ResponseParts> {
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .map_err(|e| network_error(provider, e))?;
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    Ok(ResponseParts {
        status,
        body,
        retry_after: header("retry-after").and_then(parse_retry_after),
        request_id: REQUEST_ID_HEADERS
            .iter()
            .find_map(|name| header(name))
            .map(str::to_string),
    })
}

/// Parses `Retry-After`'s delta-seconds form.
///
/// RFC 9110 permits an HTTP-date as well, and this deliberately does NOT
/// parse it: that would need a date parser, and every provider warpllm
/// speaks to sends seconds. A date form yields `None` — the header is still
/// on the response, and a caller that gets `None` waits on its own policy
/// rather than on a misparsed instant.
fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_reads_delta_seconds() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("  30 "), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
    }

    /// The date form and anything malformed are `None`, never a guess — a
    /// misparsed instant is worse than no instant.
    #[test]
    fn retry_after_declines_forms_it_cannot_read() {
        for value in ["Wed, 21 Oct 2015 07:28:00 GMT", "soon", "", "-1", "1.5"] {
            assert_eq!(parse_retry_after(value), None, "{value}");
        }
    }

    // -----------------------------------------------------------------------
    // SSE framing
    // -----------------------------------------------------------------------

    /// The sentinel these tests frame against. OpenAI-compatible providers
    /// send it; the framing itself has no opinion about the spelling.
    const DONE: &str = "[DONE]";

    /// Every payload the framing yields from one push of bytes.
    fn payloads(chunks: &[&str]) -> Vec<String> {
        let mut frames = SseFrames::new(Some(DONE));
        let mut out = Vec::new();
        for bytes in chunks {
            frames.push(bytes.as_bytes());
            while let Some(payload) = frames.next_payload().unwrap() {
                out.push(payload);
            }
        }
        out.extend(frames.flush());
        out
    }

    /// The ordinary case, one whole body at a time.
    #[test]
    fn data_lines_dispatch_on_a_blank_line_and_done_ends_the_stream() {
        assert_eq!(
            payloads(&["data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n"]),
            ["{\"a\":1}", "{\"a\":2}"],
            "the sentinel is the stream ending, never an event"
        );
    }

    /// The case no fixture can produce: a read boundary lands mid-line, and
    /// mid-character. Neither may lose or mangle a byte.
    #[test]
    fn an_event_split_across_reads_is_rejoined() {
        assert_eq!(payloads(&["data: {\"a\":", "1}\n\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data: {\"a\":1}", "\n", "\n"]), ["{\"a\":1}"]);
        // "일" is three bytes; the split falls inside it.
        let text = "data: {\"a\":\"일\"}\n\n".as_bytes();
        let (head, tail) = text.split_at(14);
        let mut frames = SseFrames::new(Some(DONE));
        frames.push(head);
        assert_eq!(frames.next_payload().unwrap(), None, "a partial character");
        frames.push(tail);
        assert_eq!(
            frames.next_payload().unwrap().as_deref(),
            Some("{\"a\":\"일\"}")
        );
    }

    /// Keepalive comments and the SSE fields warpllm does not read must not
    /// become events, and must not disturb the one being accumulated.
    ///
    /// `event:` is among them, which is what lets Anthropic — the one protocol
    /// here that names its events on that line — reuse this framing unchanged.
    #[test]
    fn comments_and_other_fields_are_ignored() {
        assert_eq!(
            payloads(&[": ping\nevent: message\nid: 7\nretry: 100\ndata: {\"a\":1}\n\n"]),
            ["{\"a\":1}"]
        );
        assert_eq!(payloads(&[": ping\n\n: ping\n\n"]), [] as [String; 0]);
    }

    /// CRLF and the optional leading space are framing, not payload.
    #[test]
    fn crlf_and_the_optional_space_are_stripped() {
        assert_eq!(payloads(&["data: {\"a\":1}\r\n\r\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data:{\"a\":1}\n\n"]), ["{\"a\":1}"]);
        // Only ONE space belongs to the framing; the rest is payload, and
        // `trim` on dispatch is what keeps it decodable either way.
        assert_eq!(payloads(&["data:  {\"a\":1}\n\n"]), ["{\"a\":1}"]);
    }

    /// The specification allows one event's data to span several lines; no
    /// provider does it today, so this pins the behaviour rather than a bug.
    #[test]
    fn multi_line_data_joins_with_newlines() {
        assert_eq!(payloads(&["data: {\"a\":\ndata: 1}\n\n"]), ["{\"a\":\n1}"]);
    }

    /// The sentinel leaves the framing in the one state a reader must not
    /// mistake for "send me more bytes": `next_payload` says `Ok(None)` to
    /// both, and only [`SseFrames::done`] tells them apart.
    ///
    /// That ambiguity is why a reader rechecks `done` before awaiting the
    /// socket. Without the recheck an upstream that holds the response open
    /// after `[DONE]` blocks the caller forever — a hang, so the symptom is
    /// untestable, but the state that causes it is exactly this.
    #[test]
    fn the_sentinel_leaves_no_payload_and_a_done_flag() {
        let mut frames = SseFrames::new(Some(DONE));
        frames.push(b"data: [DONE]\n\n");
        assert_eq!(frames.next_payload().unwrap(), None);
        assert!(frames.done, "only this distinguishes ended from starved");

        let mut starved = SseFrames::new(Some(DONE));
        starved.push(b"data: {\"a\":1}");
        assert_eq!(starved.next_payload().unwrap(), None);
        assert!(!starved.done);
    }

    /// A protocol with no sentinel gets no sentinel: the same bytes that end
    /// an OpenAI-compatible stream are an ordinary payload here, and `done`
    /// never becomes true on its own.
    ///
    /// Anthropic is that protocol, and this is why its reader has to decide
    /// for itself when the stream is over rather than asking the framing.
    #[test]
    fn without_a_sentinel_every_payload_is_an_event() {
        let mut frames = SseFrames::new(None);
        frames.push(b"data: [DONE]\n\ndata: {\"a\":1}\n\n");
        assert_eq!(frames.next_payload().unwrap().as_deref(), Some("[DONE]"));
        assert_eq!(frames.next_payload().unwrap().as_deref(), Some("{\"a\":1}"));
        assert!(!frames.done);
    }

    /// A body that ends without its final blank line still has an event.
    #[test]
    fn a_truncated_final_event_is_flushed() {
        assert_eq!(payloads(&["data: {\"a\":1}\n"]), ["{\"a\":1}"]);
        assert_eq!(payloads(&["data: [DONE]\n"]), [] as [String; 0]);
    }

    /// Invalid UTF-8 is reported rather than replaced: a lossy decode would
    /// hand the caller a chunk the provider never sent.
    #[test]
    fn invalid_utf8_in_a_line_is_an_error() {
        let mut frames = SseFrames::new(Some(DONE));
        frames.push(b"data: \xff\xfe\n\n");
        assert!(frames.next_payload().is_err());
    }
}
