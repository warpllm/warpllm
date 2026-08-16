# Changelog

Notable changes to warpllm. One version releases all three packages together —
the crate, the PyPI package, and the npm package share a version number, so a
release note here applies to all of them unless it says otherwise.

Versions follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a breaking change bumps the MINOR number: `0.1.x` and `0.2.x` are
incompatible, and `^0.1` will not upgrade you into one.

## [Unreleased]

Point warpllm at your own roster. Self-hosted OpenAI-compatible servers — vLLM,
TGI, Ollama, llama.cpp — are first-class routable targets, with no fork and no
key required.

The registry used to be compiled into the binary and nothing else, so the only
way to add a model was to fork the crate. That is disqualifying for exactly the
people running their own models: those live behind their own addresses, on their
own hardware, under names warpllm could not ship even if it wanted to.

Now a client can be handed a file in the same schema as `specs.yaml`:

```yaml
# ./warpllm.yaml
providers:
  local:
    base_url: "http://localhost:8000/v1"
    auth: none                       # the box is on a private network
    models:
      local/llama-3.3-70b:
        supported_apis:
          - {api: openai_compat_chat_completions}
          - {api: openai_compat_chat_completions_stream}
```

```python
client = WarpLLM(specs_path="./warpllm.yaml")
client.chat_completions({"model": "local/llama-3.3-70b", "messages": [...]})
```

```bash
warpllm-server --specs ./warpllm.yaml
```

Your file is **merged** over the built-in roster: adding `local/` leaves
`openai/` exactly where it was, and one client routes both. Reusing a built-in
provider's name replaces that provider whole, models included, and warpllm warns
rather than shadowing it silently — through `tracing`, so `warpllm-server` and
any Rust client with a subscriber installed surface it. The Python and Node
bindings install no subscriber, so the warning goes nowhere there; that is
already true of the older warning about an environment holding no provider keys,
and bridging `tracing` into both host languages is its own change.

It is read when the client is built, like API keys and for the same reason —
which is also where everything wrong with it is reported. A `base_url` with a
trailing slash, a model listing no surface, a provider nothing routes to: all of
these used to be things a roster could say and a request would discover. A
gateway with a bad roster now refuses to start instead of starting clean and
failing every call.

**Beside 0.4.0's `providers` declaration.** The two compose, and the composition
is the useful part rather than an accident. A provider from your own file may be
declared like any other — the declaration is checked against the roster this
client actually loaded, so `providers: {"local": {}}` is legal and narrows the
client to your box alone. And an inline `api_key` still wins over `auth: none`:
the roster says what the host wants in general, while a caller who put a token
in front of their own box has said something more specific. That is the same
precedence an inline key already had over an environment variable.

**Beside weighted load balancing.** `BalancedClient` resolves its candidates
against the roster of the client it wraps, so balancing across two boxes of your
own — the ordinary self-hosting shape — works. It previously read the shipped
roster through the free `fetch_model`, which was correct while there was only
one roster and became wrong the moment a client could carry its own: it would
have refused a caller's own entries while `chat_completions` served the very
same string. `Candidate` lost its `provider` and `model` fields in the process;
both were `#[allow(dead_code)]`, since the selected `model_str` is written back
onto the request and resolved again by the inner client.

**Why no wildcard.** `local/*` is a load error rather than a catch-all, and this
came up as a real question — the usual argument for failing closed is that a
typo becomes a live billed request, and on a box you own it is a 404 instead.
The rule stays because `supported_apis` and `capabilities` are per model: a
pattern would have to claim both on behalf of models nobody enumerated, which is
the same fails-open claim a bare `{}` entry has always been refused for. If your
server's model set moves often, generate the file from its own `/v1/models`.

### Added

- `ClientConfig::specs_path`, `WarpLLM(specs_path=…)` in Python,
  `{ specsPath }` in Node, `warpllm-server --specs <PATH>`, and `WARPLLM_SPECS`
  for all of them. Consulted in that order; an empty environment variable counts
  as unset, and a path that names nothing is an error rather than a silent
  fallback to the built-in roster.
- `Client::fetch_model`, the per-client counterpart of `warpllm::fetch_model`.
  The free function still answers about the roster warpllm **ships**, which is a
  real question and now a different one.
- `auth: none` on a provider entry: this host takes no credential, so no
  `Authorization` header is sent at all. Deliberately NOT the meaning of an
  absent `env_api_key`, which still means the roster records no way to
  authenticate the provider — otherwise a forgotten line on a paid provider
  would quietly become an unauthenticated request.
- `ProviderSpec::unauthenticated`, which tells those two apart.
- `Error::InvalidRoster`, code `invalid_roster`, rendered as a 500. It is the
  gateway's own configuration and never the caller's payload, so a 400 would
  send someone off to fix a request that was fine.
- `examples/warpllm.yaml` and `examples/self_hosted.{rs,py,ts}`, plus
  `tests/live_self_hosted.rs` — an opt-in test against a real server, since a
  mock cannot prove that a real backend's replies decode.

### Fixed

- **An OpenCode Zen account out of credit was reported as an authentication
  failure**, sending the caller to check an API key that was fine. It now
  classifies as `quota_exceeded`, which is what the failure is — and on the
  OpenAI-compatible surface that means HTTP 429 `insufficient_quota` rather
  than 401 `invalid_api_key`.

  Zen reports credit exhaustion at **HTTP 401**, not the 402 that would have
  been readable, with a family of its own and no `code` at all:

  ```json
  {"type":"error","error":{"type":"CreditsError","message":"No payment method. Add a payment method here: …/billing"}}
  ```

  The status cannot decide this on its own, because Zen answers a genuinely
  bad key with that same 401 and a different family — `AuthError` — so a rule
  reading 401 would be wrong in one direction or the other whichever way it
  went. Only the two signals together separate them.

  The classifier could not express that. It ranked every status above every
  `type`, so `openai_compat`'s reading of a bare 401 answered first and no
  provider rule written on a family was ever consulted. The two lookups are
  now one, `ErrorMapper::from_status_and_type`, and which of the two signals
  is stronger is each vocabulary's own call rather than a fixed order imposed
  on all of them — `openai_compat` and `anthropic` both still read the status
  first, and say why.

  No other provider reclassifies: nothing on the roster mapped a `type` of its
  own, so every existing failure resolves exactly as before. `ErrorMapper` is
  internal, so no package's API changes.

- A provider taking no credential was **unreachable**, not merely unsupported.
  `env_api_key` has always been optional, but such a provider was skipped when
  keys were resolved, so every request to it failed with `MissingApiKey` and
  never left the process. `auth: none` is the fix, and the bearer token is now
  attached conditionally rather than always.
- A `*` in a model key is a load error. It was read literally, so `local/*`
  registered one model named `*` and routed nothing — the one mistake a roster
  could make that looked like it had worked.
- `examples/.env.example` claims to mirror the roster and had drifted;
  `scripts/check-env-example.sh` now holds it to that, in CI, in both
  directions — a roster variable with no block here, and a block here the
  roster no longer names. It found three drifts on arrival: `MOONSHOT_API_KEY`,
  `MISTRAL_API_KEY` and `OPENCODE_API_KEY` had all shipped without one.

### Changed — breaking

Narrow, and worth stating precisely rather than leaving to imagination:

- `ClientConfig` gained a field, so an exhaustive struct literal without
  `..Default::default()` no longer compiles. **This is the only source break for
  a Rust caller** — and 0.4.0 already forced that same edit for the same reason,
  so anyone who took its advice is unaffected by this one.
- `Client::new` now reads a file when one is configured, and can fail with
  `Error::InvalidRoster`. `Error` is `non_exhaustive`, so the variant itself is
  not a break.
- `ProviderSpec::name` and `env_api_key` return `&'static str` rather than
  `&str`. Source-compatible in every ordinary use — `&'static str` coerces —
  and it is what lets a per-client roster exist without changing the public
  error types.

Python and Node are purely additive. `ProviderError`, `ChatCompletionStream`,
and every error code are untouched.

## [0.4.0] - 2026-08-16

Two additions and one source-break. A client can now declare the providers it
serves — narrowing both which environment variables are read and which models
are routable — and OpenCode Zen joins the roster as `opencode`.

The break is Rust-only and one line to fix: `ClientConfig` gained a field, so an
exhaustive struct literal no longer compiles. Python and TypeScript are purely
additive, and a Rust caller already using `..Default::default()` is unaffected.
That is what moves the MINOR number rather than the patch, per the versioning
note at the top of this file — `^0.3` will not upgrade you into it.

### Added

- A client can declare the providers it serves. `ClientConfig.providers` in
  Rust, `providers=` in Python, `providers:` in TypeScript — a map keyed by
  registry name, each entry optionally carrying an `api_key`.

  ```python
  WarpLLM(providers={"openai": {}, "deepseek": {"api_key": "sk-..."}})
  ```

  It narrows two things at once. Only the declared providers' environment
  variables are read, so a key exported for something else is not quietly
  adopted; and only the declared providers are routable, so a request for a
  model under one you did not name is refused before any upstream call, with a
  new `provider_not_declared` error rather than the missing-credential error
  that would have sent you after a key you deliberately withheld.

  An inline `api_key` is for callers holding keys somewhere the process
  environment cannot reach — a secret manager, a per-tenant database row. It
  wins over the variable the roster names, and it can authenticate a provider
  whose roster entry names no variable at all, which was previously impossible.

  A provider name the roster does not hold fails when the client is built, not
  at the request that happened to route there.

- **OpenCode Zen**, a new `opencode` provider reached at `opencode.ai/zen/v1`
  with `OPENCODE_API_KEY`, serving sixteen models: `deepseek-v4-pro` and
  `-flash`, `glm-5.1` and `glm-5.2`, `minimax-m2.7` and `m3`, Kimi K2.6
  through K3, Zen's own `big-pickle`, and the six `-free` models it bills at
  nothing during their evaluation window.

  Zen is the first provider here that does not serve its whole catalog over one
  protocol. It publishes an endpoint PER MODEL — `/chat/completions`,
  `/responses`, `/messages`, or Google's `/models/<id>` — and only the first is
  a surface warpllm implements, so that is what this provider is.

  **Zen's GPT models are deliberately absent**, and so are Grok and Muse: all
  twenty-four sit on `/responses`, the Responses API, which warpllm does not
  implement yet. Registering them would hand back a model string that resolves
  and then fails upstream on every request, so they wait for that surface —
  the same reason `openai/gpt-5-pro` has never been on the roster. Zen's Claude
  and Qwen models (`/messages`) and its Gemini models (`/models/<id>`) are out
  for the same kind of reason: neither protocol has an `api:` here at all.

  `glm-5`, `kimi-k2.5` and `minimax-m2.5` are absent although Zen serves all
  three on chat completions and still returns them from `GET /zen/v1/models`:
  its DEPRECATED MODELS section retired them on 2026-05-14, 2026-08-05 and
  2026-08-05, every date already past. The live successors — `glm-5.2`,
  `kimi-k3`, `minimax-m3` — are on the roster instead.

  Nothing states an endpoint but that docs table, so a Zen model added to the
  wrong one would lint clean and bill for a request that reaches nothing.
  `no_opencode_entry_sits_on_a_surface_warpllm_cannot_reach` is the gate, and
  it matches on family prefix so a name Zen adds later is caught without the
  test being edited.

  No `capabilities` on any entry: Zen publishes a price per model and no
  context window, output ceiling, or concurrency figure. Every model is
  re-hosted, so the original provider's numbers are not Zen's to promise.

  The streaming surface was registered on inference — Zen documents no request
  parameter at all, and prescribes `@ai-sdk/openai-compatible` as the client
  for exactly these models — and then checked against the live endpoint, which
  returns OpenAI-compatible SSE chunks that survive the whole warpllm path.
  `live_stream.rs` carries a Zen row so it stays checked.

### Changed

- **Source-breaking for Rust only.** `ClientConfig` gained a field, so an
  exhaustive struct literal no longer compiles; add `providers: None` or switch
  to `..Default::default()`. Nothing else changes: a client that leaves
  `providers` alone behaves exactly as before, reading the whole roster's
  variables and routing to all of it. Python and TypeScript are purely
  additive — the new argument is optional in both.

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
