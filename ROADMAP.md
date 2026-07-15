# Roadmap

warpllm broke ground on **July 9, 2026** (initial commit [`cd10f76`](https://github.com/warpllm/warpllm/commit/cd10f76a4928c631d588a166bff0749fef4b73e6)), so this roadmap is young and deliberately coarse — quarters, not dates. It is a statement of direction, not a delivery contract, and it is meant to be shaped in the open: if an item below matters to you (or is missing), say so on [Discord](https://discord.gg/tSSQTxFnsC) or open an issue.

The destination: a self-hosted, Rust-fast gateway you can put in front of every LLM provider you use — one OpenAI-compatible surface with the reliability features (failover, load balancing, caching, metrics) that production traffic demands.

## Shipped

- Rust core with **OpenAI chat completions** (non-streaming): `provider/model` routing strings, env-var key pickup, typed errors.
- **Exact OpenAI `chat.completion` response shape** — a field-for-field copy of the upstream API, enforced by fixture round-trip tests and SDK-parity checks against the official OpenAI Python and Node SDKs.
- **Python and Node bindings** over the shared Rust core (PyO3 / napi-rs) with idiomatic typed clients and exceptions — currently built from source.

## Now — Q3 2026

- **Streaming** chat completions (SSE), in Rust and both bindings.
- **Publish the bindings**: `pip install warpllm` and `npm install warpllm` with prebuilt wheels/binaries — no Rust toolchain required.
- **OpenAI-compatible providers** — DeepSeek, Mistral, Groq, Together, and friends. The core is already structured so these reuse the OpenAI wire types and only carry their own quirks.
- **Anthropic** — the first fully translated provider (its own wire format in, OpenAI shape out).
- **More OpenAI endpoints** beyond chat: embeddings and moderations first.

## Next — Q4 2026

- **Remaining major providers**: Google (Gemini), Cohere, Amazon (Nova/Bedrock), Meta Llama via its hosting providers — heading toward broad model-catalog coverage.
- **Automatic failover** — fallback chains across providers/models, with retry and backoff policy.
- **Load balancing** — weighted distribution across providers, regions, and API keys.
- **Metrics** — token usage, cost, latency, and error rates per provider/model, exportable to your observability stack.
- **Response caching** — configurable TTL and cache keys.

## Later — 2027

- **Smart routing** — automatic provider/model selection on cost, latency, and availability, built on the metrics layer.
- **More endpoints**: images, audio (speech + transcription), batch.
- **Standalone server mode** — an OpenAI-compatible HTTP proxy for polyglot stacks that can't embed the library. Library-first remains the identity; the server wraps the same core.
