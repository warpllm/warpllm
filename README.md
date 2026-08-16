<div align="center">

# warpllm

A warp-speed, robust AI gateway written for rust, node, and python applications - built for planet scale by the community.

[![Discord](https://img.shields.io/badge/Discord-warpllm-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/payU7Vzuq5)
[![Reddit](https://img.shields.io/badge/Reddit-r%2Fwarpllm-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/r/warpllm/)

[![CI](https://img.shields.io/github/actions/workflow/status/warpllm/warpllm/ci.yml?branch=main&logo=github&label=CI)](https://github.com/warpllm/warpllm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/warpllm?logo=rust&label=crates.io)](https://crates.io/crates/warpllm)
[![PyPI](https://img.shields.io/pypi/v/warpllm?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/warpllm/)
[![npm](https://img.shields.io/npm/v/%40warpllm%2Fwarpllm?logo=npm&label=npm)](https://www.npmjs.com/package/@warpllm/warpllm)
[![License](https://img.shields.io/github/license/warpllm/warpllm?label=license)](https://github.com/warpllm/warpllm/blob/main/LICENSE)

</div>

## Quickstart

```bash
pip install warpllm              # python
npm install @warpllm/warpllm     # node
cargo add warpllm                # rust
```

```bash
export OPENAI_API_KEY=sk-...
```

**Python**

```python
from warpllm import WarpLLM

client = WarpLLM()

completion = client.chat_completions({
    "model": "openai/gpt-5-nano",
    "messages": [{"role": "user", "content": "Hello!"}],
})

print(completion["choices"][0]["message"]["content"])
```

**Node**

```ts
import { WarpLLM } from '@warpllm/warpllm'

const client = new WarpLLM()

const completion = await client.chatCompletions({
  model: 'openai/gpt-5-nano',
  messages: [{ role: 'user', content: 'Hello!' }],
})

console.log(completion.choices[0].message.content)
```

**Rust** — `chat_completions` is `async` and warpllm ships no runtime, so bring
your own: `cargo add tokio --features macros,rt-multi-thread`.

```rust
use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ClientConfig::default())?;

    let completion = client
        .chat_completions(CreateChatCompletionRequest {
            model: "openai/gpt-5-nano".to_string(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        })
        .await?;

    let content = completion.choices[0].message.content.as_deref();
    println!("{}", content.unwrap_or_default());
    Ok(())
}
```

### Switching providers is a string change

Keys are read from the environment when the client is built, so export the one
the provider needs and change the model string. Nothing else moves.

| Model string | Key it needs |
| --- | --- |
| `openai/gpt-5-nano` | `OPENAI_API_KEY` |
| `deepseek/deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `kimi/kimi-k3` | `MOONSHOT_API_KEY` |
| `opencode/glm-5.2` | `OPENCODE_API_KEY` |
| `openrouter/anthropic/claude-sonnet-4` | `OPENROUTER_API_KEY` |

The `provider/` prefix is required. warpllm matches the whole string against its
roster, so a bare `gpt-5-nano` — or any name it doesn't know — is an error
rather than a guess at an upstream default.

### Narrowing a client to the providers it serves

By default a client serves the whole roster and reads every provider's variable.
Declare the ones you mean and it reads no others, routes to no others, and takes
a key directly for the callers who keep theirs somewhere the environment can't
reach:

```python
WarpLLM(providers={"openai": {}, "deepseek": {"api_key": "sk-..."}})
```

```ts
new WarpLLM({ providers: { openai: {}, deepseek: { apiKey: 'sk-...' } } })
```

```rust
ClientConfig {
    providers: Some(BTreeMap::from([
        ("openai".into(), ProviderConfig::default()),
        ("deepseek".into(), ProviderConfig { api_key: Some(key) }),
    ])),
    ..Default::default()
}
```

An empty entry means "serve this one, key from the environment". A request for a
model under a provider you didn't declare is refused before any upstream call,
and a provider name the roster doesn't hold fails when the client is built.

Runnable versions of all three, with comments, are in
[`examples/`](examples/).

## Mission

This project is to lay out the most resilient open source productionization layer for AI-deployments. Designed for you if you want:

1.  To work with multiple AI providers or your own models.
1.  To keep your AI services up and running with 0 downtime.
1.  Speed (minimal overhead latency).
1.  A granular view of your metrics (uptime, P95 latency, costs, etc).
1.  Control over:
    1.  Where your data goes.
    1.  Your AI budget across providers.

## Status

> [!IMPORTANT]
> The published packages are **0.3.1**, which adds streaming — from the Rust
> core, both SDKs, and the HTTP gateway — alongside the Kimi provider and the
> rest of OpenAI's chat-completion roster. It is a **breaking** release:
> response fields that are optional *and* nullable now tell an absent key from
> an explicit `null`, so their type changed in all three languages. See the
> [changelog](CHANGELOG.md) before upgrading from `0.2.x`.
>
> The OpenAI-compatible HTTP gateway has landed on `main` but is **not
> released yet**.

| | Released (0.3.1) | On `main` |
| --- | --- | --- |
| OpenAI chat completions, non-streaming | Yes | Yes |
| `provider/model` routing strings | Provider registry | Provider registry |
| DeepSeek, OpenRouter | Yes | Yes |
| Kimi | Yes | Yes |
| OpenAI-compatible HTTP gateway | — | Unreleased |
| Streaming | Yes | Yes |
| Failover, load balancing, caching, metrics | — | — |

Unlisted models are rejected rather than guessed at, so routing a name warpllm
doesn't know is an error, not a surprise upstream bill.

## Layers

1.  **An SDK** - provide a request and we translate it to work with different providers and models out of box.
1.  [Unreleased] **A proxy** - run a self-hosted proxy that speaks the OpenAI API:
    1.  [Coming Soon] **Failover** - define multiple models to handle outages / errors
    1.  [Coming Soon] **Load Balancing** - define a % of requests to be handled per model
    1.  [Coming Soon] **Prompt Response Caching** - define a TTL and avoid paying twice for the same prompt

## Key focus points

1.  **Native SDK support** - Written once in rust, compiled for maximum performance, available for rust/typescript/python.
1.  **Self hostable** - Avoid vendor lock-in (e.g. from cloud provider or model provider), or data leaving your infra.
1.  **Warp-speed execution** - What we named ourselves after. Machine level code, faster than a typescript or python native library.
1.  **Compact file size** - Pre-compiled into binary format, not verbose text files.

## Roadmap

The roadmap lives in [GitHub issues](https://github.com/warpllm/warpllm/issues) — one issue per item, so direction is discussed where the work happens. Add a comment if you see something missing, or if something there matters enough to you that it should move up.

## Contributing

We're excited to have you join us. See the
[contribution guide](CONTRIBUTING.md) for how to get started.

A big thank you to the contributors below who have helped build this AI gateway to this point!

<a href="https://github.com/warpllm/warpllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=warpllm/warpllm" />
</a>

## License

The warpllm core is open source under the [Apache License 2.0](https://github.com/warpllm/warpllm/blob/main/LICENSE).
