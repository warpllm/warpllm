# Changelog

Notable changes to warpllm. One version releases all three packages together —
the crate, the PyPI package, and the npm package share a version number, so a
release note here applies to all of them unless it says otherwise.

Versions follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a breaking change bumps the MINOR number: `0.1.x` and `0.2.x` are
incompatible, and `^0.1` will not upgrade you into one.

## [0.3.1] - 2026-08-13

Nothing in any of the three packages changed. 0.3.0 reached crates.io and PyPI
and did not reach npm: the npm workflow runs the Node suite on Windows before
publishing, and one test read the recorded SSE transcripts by splitting on `\n`
alone. A Windows checkout stores those fixtures with CRLF, which left a `\r` on
every payload and stopped the `[DONE]` sentinel from matching — three tests
failed against a corpus the Rust suite reads without complaint.

So this release is 0.3.0 for npm, where it is the first version carrying
streaming, and 0.3.0 with a test fix everywhere else. Read the 0.3.0 notes below
for what actually landed; upgrading from npm `0.2.0` is the breaking step those
notes describe.

### Fixed

- The Node stream-consumer test splits on `\r?\n`, which is what SSE terminates
  a line with and what Rust's `str::lines()` already handled.
- `.gitattributes` holds `*.sse` fixtures to LF on every platform, so all three
  language suites read the corpus as it was recorded.

## [0.3.0] - 2026-08-13

The release that adds streaming. 0.2.0 could only hand back a whole reply; this
streams chat completions from every surface — the Rust core, both SDKs, and the
OpenAI-compatible gateway.

Everything under "Changed" is breaking.

### Added

- **Streaming from every surface: the SDKs and the HTTP gateway.** The Rust core
  streamed; nothing else could reach it. Now all of them do.

  TypeScript and Python get it on the method they already call, the way the
  official OpenAI packages do it:

  ```ts
  for await (const chunk of await client.chatCompletions({ ...req, stream: true })) {
    process.stdout.write(chunk.choices[0]?.delta.content ?? '')
  }
  ```

  ```python
  for chunk in client.chat_completions({**req, "stream": True}):
      print(chunk["choices"][0]["delta"].get("content", ""), end="")
  # ...and `async for` on AsyncWarpLLM.
  ```

  Python's static typing is the one asymmetry, and it is a consequence of taking
  one mapping rather than `**kwargs` so unmodeled fields cross untouched: an
  overload has to match the mapping's type, and a dict carrying extra keys
  matches no `TypedDict`. The runtime behaviour is exact either way, and
  `chat_completions_stream` is there for callers who need a checker to agree.
  TypeScript needs no such hatch.

  `warpllm-server` serves `stream: true` as Server-Sent Events, so an OpenAI SDK
  pointed at the gateway now gets a stream instead of something it cannot parse.
  A refusal before the first chunk keeps its real status, `Retry-After` and all;
  once the 200 is committed there is no status left, so a failure arrives as a
  final `data: {"error": …}` event with **no** `[DONE]` after it — a caller has
  to be able to tell a truncated answer from a complete one.

  Underneath, `JsonClient` gained `chat_completions_stream` and `JsonChatStream`.
  It stays free of any iterator trait: Rust has several and the languages on the
  far side have their own, so an inherent `next` is what each binding builds
  `Symbol.asyncIterator` or `__anext__` on.

- **`stream_read_timeout_secs`**, bounding how long a stream may go without a
  single byte (`streamReadTimeout` in Node, `stream_read_timeout` in Python).
  Absent means never, which is the default and today's behaviour.

  `timeout_secs` is a TOTAL deadline: it cannot tell a stream that is alive and
  slow from one that is wedged, so it bounds a stall only by outliving it — and
  it cuts off a healthy long generation at the same mark. This bounds the GAP
  between reads instead and resets on every byte, which is the shape that fits a
  response whose length nobody knows in advance. A stream that goes quiet past
  the limit ends with `Error::StreamStalled`, a 504 naming the limit that fired.

  Opt-in because no single value is right for everyone, and a wrong one fails in
  the worst direction: the wait before the FIRST chunk is a gap like any other,
  and a reasoning model can think for minutes before it emits a token. Set it
  above the slowest time-to-first-token you expect, not merely above the gap
  between chunks.

  It is applied per-stream rather than through reqwest's own `read_timeout`,
  which is builder-scoped and would either bind every non-streamed request or
  cost a second connection pool to avoid that.

- **`scripts/test-all.sh`**, running the Rust, Node and Python suites in one
  command. `cargo test --workspace` covers one of the three, and both bindings
  are rebuilt rather than assumed current — `uv sync --locked` audits without
  recompiling, so pytest will pass against a stale extension module.

- **Streaming chat completions.** `Client::chat_completions_stream` returns a
  `ChatCompletionStream` of `CreateChatCompletionStreamResponse` chunks, read
  off the wire as they arrive:

  ```rust
  let mut stream = client.chat_completions_stream(request).await?;
  while let Some(chunk) = stream.next().await {
      for choice in &chunk?.choices {
          if let Some(Some(text)) = &choice.delta.content {
              print!("{text}");
          }
      }
  }
  ```

  The chunk shape already shipped; what is new is everything that produces one.
  Server-sent events are framed in the protocol layer — partial lines and
  multi-byte characters split across reads are rejoined, `:` keepalive comments
  are ignored, and `[DONE]` ends the stream rather than arriving as a chunk.
  An upstream that stops WITHOUT sending `[DONE]` is a truncated answer, not a
  finished one, and ends the stream with `Error::StreamTruncated` — everything
  that did arrive first. A stream has two ways to stop and only one of them
  means the reply is whole; collapsing them would hand a caller half an answer
  with nothing to say so.
  Each event is then normalized through a gateway representation and rendered
  back, losslessly: an explicit `null` returns as a `null`, an absent key stays
  absent, an opening `"content": ""` survives as itself, and per-chunk fields no
  specification names — OpenAI's `obfuscation` among them — reach the caller
  verbatim. Both recorded transcripts are checked event by event through that
  round trip.

  Every model on the roster now declares `openai_compat_chat_completions_stream`
  alongside its whole-reply surface, each provider's `stream` parameter cited in
  `specs.yaml`. Streaming remains its own surface, so a future model that serves
  one without the other still says so.

  `Client::chat_completions` refuses `stream: true` rather than serving the
  wrong shape, naming the entrypoint that serves it. `JsonClient` does the same,
  since a whole reply is the only thing its `String` can carry.

- **Kimi, and the rest of OpenAI's chat-completion roster.** A new `kimi`
  provider, reached at `api.moonshot.ai` with `MOONSHOT_API_KEY`, serving
  `kimi/kimi-k3`, `kimi/kimi-k2.6`, and the two `kimi-k2.7-code` variants —
  plus `kimi-k3`, `kimi-k2.6` and `kimi-k2.7-code` through OpenRouter, added
  to that provider's curated set. On the OpenAI side the roster grows from
  five models to eighteen: the 4.1 and 4o families, GPT-5 through 5.5, and
  o3.

  Every entry was checked against the provider's own model page for whether it
  actually serves chat completions, and its context window taken from wherever
  the provider publishes an exact figure — for Kimi that is the pricing page,
  since the model list rounds to "1M" and "256k". OpenRouter entries record
  the narrowest limit any endpoint behind the slug offers, since a slug fans
  out across every host serving those weights and an unpinned request may land
  on any of them. Models that serve only OpenAI's Responses API, that are
  gated, or that the provider has already scheduled for retirement are
  deliberately absent, and `specs.yaml` records which and why rather than
  leaving the gap to be guessed at.

- **`deprecation_date` on a model entry**, `YYYY-MM-DD`, recording the day a
  provider stops serving a model, and readable in Rust through
  `ModelSpec::deprecation_date()`. Nothing acts on it: routing does not consult
  it, and the loader takes the string as written without checking it against a
  calendar.

- **`CreateChatCompletionStreamResponse`**, the reply shape for a chat
  completion requested with `stream: true`, in all three languages. The request
  is unchanged — `CreateChatCompletionRequest` already carries `stream` — but
  upstream the reply is a separate type rather than the whole completion with
  fields left empty, and warpllm models it the same way: a choice carries a
  `delta` instead of a `message`, `finish_reason` is null until the last chunk,
  `usage` is present-but-null on every chunk before it, and a tool call arrives
  as a fragment keyed by `index` whose `arguments` are split across chunks.
  Keeping the two types apart is what lets a caller tell a fragment from a
  finished thing.

  No transport yet: nothing in the SDK returns one of these. This settles the
  shape, and its generated Python and Node types, so the client work has
  something fixed to build against.

### Fixed

- **`openrouter/anthropic/claude-sonnet-4` recorded a 1,000,000 token context
  window it could not guarantee**, and now records 200,000. The slug is served
  by two platforms: Vertex offers the larger window, Bedrock 200,000, and an
  unpinned request may be routed to either. A caller sizing a prompt against
  the old figure could have it rejected upstream.

  This is the general hazard with an aggregator, and `specs.yaml` now says so
  where the OpenRouter entries live: take limits from the per-slug
  `endpoints` listing and record the narrowest, because one slug fans out
  across every host serving it.

### Changed

- **Response fields that are optional *and* nullable now tell an absent key
  from an explicit `null`.** Seven fields on the non-streaming completion
  change type: `moderation` and `service_tier` on the response, `logprobs` on a
  choice, `content` and `refusal` on that choice's logprobs, and `audio` and
  `function_call` on a message.

  Previously a provider that sent `"logprobs": null` was read as having omitted
  the key, and warpllm re-emitted it omitted. The two states mean different
  things — for a chunk's `usage`, upstream documents absent as "you never asked
  for usage" and null as "you did, and this is not the last chunk" — so the
  round trip now preserves whichever one the provider sent.

  What this looks like per language: in Rust the field becomes
  `Option<Option<T>>`, so matching on it stops compiling; in TypeScript and
  Python the value type gains `| null` / `| None`, so a caller who assumed
  non-null gets a type error; and in the JSON both bindings hand back, a key
  that used to be dropped now appears as `null` whenever the provider sent it.

## [0.2.0] - 2026-08-09

The first release with a provider registry. 0.1.4 could reach OpenAI; this can
route to any provider on the roster, and it decides at construction which of
them the environment can actually authenticate.

This release rewrites most of the public surface. Everything under "Changed"
and "Removed" is breaking.

### Added

- **Provider registry.** Providers and models live in `specs.yaml`, compiled
  into the binary. Model strings are `provider/model`. The registry **fails
  closed**: a name no entry claims is an error, never a guess at an upstream
  default, so a typo cannot become a live, billed request. There is no
  wildcard — `openai/*` registers a model literally named `*`.
- **Every model declares the API surfaces it serves**, and inherits nothing
  from its provider:

  ```yaml
  openai/gpt-5.6:
    supported_apis:
      - {api: openai_compat_chat_completions}
  ```

  A provider is a host, not a capability — one host commonly serves chat
  completions, embeddings, and moderation from disjoint sets of models — so
  there is nothing at that level to route on. A request for a surface the model
  does not list is refused before the network, rather than discovered as a 404
  upstream. A surface name carries the protocol it is spoken in, which is what
  lets a model one day list `anthropic_messages` beside the entry above.
- **Provider entries are transport only**: `base_url`, `env_api_key`, `models`.
  There is no `protocol:` field — the surfaces a model lists already say which
  wire format is in play, so one host may serve models over different
  protocols.
- **DeepSeek and OpenRouter providers**, alongside OpenAI. Adding an
  OpenAI-compatible provider is a YAML edit and no Rust.
- **Environment-driven provider discovery.** Building a client reads each
  roster provider's `env_api_key` variable once and keeps the providers it can
  authenticate. The set is reported through `tracing` — silent unless the host
  installs a subscriber. A request is admitted only when the roster registers
  the model *and* the client holds a key for the provider serving it.
- **Error taxonomy with provenance.** `Error::origin()` separates a warpllm
  rejection from a provider's, and `Error::code()` is a stable slug for
  bindings. Provider failures carry `ProviderError` with the upstream status,
  `retry_after`, and request id.
- **Errors normalized into OpenAI's vocabulary**, once, in Rust — so a quota
  exhaustion reads the same whichever provider served it. Python and Node raise
  exception classes mirroring the official OpenAI SDK.
- Registry read surface in Rust: `fetch_model`, `ProviderSpec`, `ModelSpec`,
  `Capabilities`, `SupportedApi`, `Api`. Every field is private and there is no
  public constructor, so a spec is read-only outside the crate.
- `JsonClient`, the JSON boundary both native bindings share.
- Quickstart examples for all three languages in `examples/`.

### Changed

- **`chat_completion` is now `chat_completions`** (`chatCompletion` →
  `chatCompletions` in Node). This is the rename most callers will hit.
- **API keys resolve at construction, not per request.** A key exported after a
  client is built is not picked up, and a rotated key needs a new client. Long
  running processes that build one client at startup must restart to pick up a
  rotated key.
- **Rust wire types moved** from `types::openai::chat::completions::*` to
  `protocol::openai_compat::chat_completions::types`. The crate root no longer
  re-exports them with a glob; it names the three types you need to make a call
  and hold its result: `CreateChatCompletionRequest`,
  `ChatCompletionRequestMessage`, `CreateChatCompletionResponse`.
- **Binding types are generated from Rust** rather than hand-written, so the
  three languages cannot drift. Python's `ChatCompletion` is now
  `CreateChatCompletionResponse`; Node re-exports the generated names under
  OpenAI's spellings.
- Unknown request and response fields pass through in both directions rather
  than being dropped, so a provider parameter warpllm does not model still
  reaches it.

### Removed

- **`echo`** from every package. It was a connectivity probe, not API.
- **`WarpLLMError`, `InvalidRequestError`, `NotImplementedError`** in Python and
  Node, replaced by the OpenAI-SDK-shaped hierarchy (`APIError`,
  `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`,
  `NotFoundError`, `ConflictError`, `UnprocessableEntityError`,
  `RateLimitError`, `InternalServerError`, `APIConnectionError`).
- Python's re-exports of response internals (`Choice`, `CompletionUsage`,
  `Annotation`, and the rest). Reading `completion["choices"][0]` names none of
  them, so they are no longer part of the public surface.

### Not in this release

Streaming, retries, failover, load balancing, and caching are still
unimplemented. Supplying API keys through client configuration rather than the
environment is not supported. The OpenAI-compatible HTTP gateway
(`warpllm-server`) is in the repository but is not published.

## [0.1.4] and earlier

Early SDK releases serving OpenAI chat completions only, before the provider
registry existed. See the [release tags](https://github.com/warpllm/warpllm/tags).

[0.3.1]: https://github.com/warpllm/warpllm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/warpllm/warpllm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/warpllm/warpllm/compare/v0.1.4...v0.2.0
