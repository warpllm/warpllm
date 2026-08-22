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

### Your own models

Anything that speaks the OpenAI API — vLLM, TGI, Ollama, llama.cpp — is a
routing target. Describe it in a file and hand warpllm the path:

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

```typescript
const client = new WarpLLM({ specsPath: './warpllm.yaml' })
```

```rust
let client = Client::new(ClientConfig {
    specs_path: Some("./warpllm.yaml".into()),
    ..Default::default()
})?;
```

```bash
warpllm-server --specs ./warpllm.yaml   # or WARPLLM_SPECS=./warpllm.yaml
```

Your file is **merged** over the built-in roster, so adding `local/` leaves
`openai/` exactly where it was — the same client routes both. Reusing a
built-in provider's name replaces that provider whole, and warpllm warns rather
than shadowing it quietly. The warning goes through [`tracing`], which
`warpllm-server` surfaces and a Rust client does once it installs a subscriber;
the Python and Node bindings install none yet, so there it goes nowhere. Same
for the older warning about an environment with no provider keys in it.

[`tracing`]: https://docs.rs/tracing

`auth: none` is the line that matters for a private box: warpllm then sends no
`Authorization` header at all. Omitting it means something else — that the
roster records no way to authenticate this provider — so a forgotten
`env_api_key` on a paid provider fails locally instead of leaving without a
credential.

The file is read when the client is built, so a roster that can't be used is an
error there, naming the path — not a request failing hours later. There is no
wildcard: every model gets an entry, because `supported_apis` and
`capabilities` are per model and a pattern would have to claim both on behalf
of models nobody listed.

The schema is documented in full at the top of
[`specs.yaml`](crates/warpllm/src/registry/specs.yaml), and
[`examples/warpllm.yaml`](examples/warpllm.yaml) is a worked one covering vLLM,
Ollama, and a cluster that does want a key.

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
> The published packages are **0.4.0**, which adds the OpenCode Zen provider
> and lets a client declare the providers it serves. It is **source-breaking
> for Rust only**: `ClientConfig` gained a field, so an exhaustive struct
> literal no longer compiles — add `providers: None`, or switch to
> `..Default::default()`. Python and TypeScript are purely additive. See the
> [changelog](CHANGELOG.md) before upgrading from `0.3.x`.
>
> The OpenAI-compatible HTTP gateway has landed on `main` but is **not
> released yet**.

| | Released (0.4.0) | On `main` |
| --- | --- | --- |
| OpenAI chat completions, non-streaming | Yes | Yes |
| `provider/model` routing strings | Provider registry | Provider registry |
| DeepSeek, OpenRouter | Yes | Yes |
| Kimi | Yes | Yes |
| OpenCode Zen | Yes | Yes |
| Declaring the providers a client serves | Yes | Yes |
| Self-hosted models via your own roster file | — | Unreleased |
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
