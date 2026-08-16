# Contributing to warpllm

Thank you for your interest in contributing to warpllm. Welcome to the community building a warp speed AI gateway.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment:

* Be respectful and constructive in all interactions
* Welcome newcomers and help them get started
* Focus on what's best for the community and project
* Accept constructive criticism gracefully
* Show empathy towards other community members
* Leave your ego at the door — take help when you need it, and offer it when you
  see someone asking

Report any unacceptable behavior to us through our [Discord](https://discord.gg/payU7Vzuq5).

## How to contribute

### Prerequisites

You need Git and Rust + cargo for any contribution — the core is Rust, and the
bindings compile it too. Add the others only for the package you're touching:

* **Git** and **Rust + cargo** — always
* **Python 3.10+ and uv** — only for `bindings/python`
* **Node 22+ and npm** — only for `bindings/node`
* **A signed [CLA](https://cla-assistant.io/warpllm/warpllm)** — we can't merge
  your PR without it, so sign it before you start rather than at review time.
  Full text in [CLA.md](CLA.md).

### Steps

Every change starts as an issue, and one person owns it at a time.

1.  **Pick something to work on.** We welcome contributions in several areas:
    * **Model/provider integrations**: Improve the AI Gateway by maintaining or adding more models and providers. Usually an edit to `registry/specs.yaml` — see [Adding a provider or model](#adding-a-provider-or-model).
    * **Adding protocols**: Sometimes we see new protocols outside of the ones we support (erhm.. OpenAI-Compatible API). These live in `protocol/`, with conversions in `gateway/`.
    * **Documentation**: Improve guides, examples, and API docs
    * **Testing**: Increase test coverage always helps
    * **Examples**: Create demos and use cases on how to use warpllm
    * **Bug Fixes**: Fix reported issues
    * **Performance**: Simplify code, reduce latency, or reduce memory usage
1.  **Search first.** Look through
    [open and closed issues](https://github.com/warpllm/warpllm/issues?q=is%3Aissue)
    before writing anything up. If it's a new feature idea and you want opinions
    on the shape, raise it in `#dev` on
    [Discord](https://discord.gg/payU7Vzuq5) first.
1.  **File it** if it doesn't exist yet, using the
    [issue templates](https://github.com/warpllm/warpllm/issues/new/choose).
1.  **Claim it.** Comment **"working on this"** and a maintainer assigns you.
    The assignment is the signal — don't start on an issue already assigned to
    someone else. On your own feature request, ticking *I'd like to open the PR*
    is the claim.
1.  **Build it.** [Development Setup](#development-setup), then
    [Before you open a PR](#before-you-open-a-pr) for the gate CI runs.
1.  **Open the PR**, referencing the issue (`Fixes #123`).
1.  **De-claim if you're stuck.** Two days past where you expected to be with no
    path forward, or the issue is blocking other work? Post what you have and
    where it broke, in the thread or `#dev`. Ask for help or hand it back —
    both are fine, and neither is a failure. Silence is the only thing that
    hurts. Claims with no comment and no PR for two weeks get reopened to
    everyone.

## Project structure

warpllm is a single Cargo workspace. The Rust core does the work; the Python and
Node packages are thin bindings over that same core, so a fix in `crates/warpllm`
reaches all three languages at once.

```
crates/warpllm/          The SDK. Everything below is a module of this crate.
  registry/              Which providers and models exist.
    specs.yaml           The roster itself — adding a model is an edit here.
  protocol/              Wire shapes: what a provider's API actually sends
                         and receives, per wire format (not per provider).
  gateway/               warpllm's own request/response types, and the
                         conversions between them and the wire shapes.
  types.rs               `Api`: which surface a model serves, named so that it
                         also says which wire format that surface is spoken in.
  client.rs              Routes a request: look up the model, check it serves
                         the surface, send it, convert the response back.
crates/warpllm-server/   The OpenAI-compatible HTTP gateway (unreleased),
                         an axum server wrapping the SDK.
bindings/python/         PyO3 + maturin. Rust glue in src/, the importable
                         package in python/warpllm/, tests in tests/.
  python/warpllm/_generated/   Generated. Do not edit.
bindings/node/           napi-rs. Rust glue in src/, TypeScript in src-ts/,
                         tests in __test__/.
  src-ts/generated/      Generated. Do not edit.
examples/                One quickstart per language, side by side.
                         crates/warpllm/examples/quickstart.rs is a SYMLINK to
                         the file here — cargo only finds examples under the
                         package root, and the three should not drift apart.
                         Edit the real file; do not replace the link with a copy.
```

The three ideas worth knowing before you read the code:

* **The registry is the roster, and it fails closed.** A model warpllm doesn't
  know is an error, never a guess at some upstream default. The header comment
  in [`specs.yaml`](crates/warpllm/src/registry/specs.yaml) explains the
  provider/model split and the rules the lint enforces — read it before adding
  either.
* **A surface names its own protocol, and the model names the surface.** An
  `Api` is spelled `<protocol>_<endpoint>` — `openai_compat_chat_completions` —
  so a model's `supported_apis` is the only place the roster records a wire
  format. A provider entry is transport alone, which is what lets one host
  serve models over different protocols. Providers on the same protocol share
  its module, and one that diverges states only its delta under that protocol's
  `provider_overrides/`; a provider that matches it implements nothing and
  inherits it whole, which is why adding an OpenAI-compatible provider is
  usually a registry edit and no new Rust. A backend whose wire SHAPE differs
  is a new protocol, not an override.
* **The bindings hold no wire shapes of their own.** The non-published
  `warpllm-codegen` workspace tool emits their generated request, response, and
  error types from Rust. Small handwritten facade files decide which names are
  public; they do not repeat any fields.
* **The official OpenAI SDKs are oracles, never contracts.** Both bindings
  carry `openai` as a pinned dev dependency and compare warpllm's shapes
  against it — TypeScript by assignability, Python by walking the pydantic
  models' field names. Nothing is re-exported and neither published package
  depends on it: the checks say warpllm still fits what the vendor documents,
  and fail when it stops. A deviation is allowed, but it gets written down at
  the check rather than discovered in someone's stream.

## Development Setup

Clone the repo, then verify the toolchain works before changing anything. The
Rust toolchain is pinned by `rust-toolchain.toml`, so `cargo` installs the right
version on first use.

```bash
git clone https://github.com/warpllm/warpllm.git
cd warpllm
cargo test --workspace
```

The bindings each build a native module from the Rust core, so they need a
working `cargo` too — but you only need to set one up if you're changing that
language's package:

```bash
# Python
cd bindings/python && uv sync --locked && uv run pytest

# Node
cd bindings/node && npm ci && ./node_modules/.bin/napi build --platform && npm test
```

**No API keys are needed to develop or run the tests.** The suites run against
mock HTTP servers. Keys are only read at request time, from the routed
provider's environment variable (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, …), if
you want to make a real call by hand.

One test asks for keys and is `#[ignore]`d for exactly that reason: it streams
a short completion from every provider whose key is set and checks that each
chunk survives warpllm's shapes verbatim. Recorded fixtures cannot notice a
provider changing what it sends, so run it when you have keys around, and turn
anything it finds into a fixture under
`crates/warpllm/tests/protocol/openai_compat/chat_completions/fixtures/transcript/`:

```bash
OPENAI_API_KEY=... cargo test -p warpllm --test live_stream -- --ignored --nocapture
```

### Before you open a PR

These are exactly what CI runs, so running them locally is the whole gate:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Plus the suite for any binding you touched, from the commands above. Tests are
part of the change, not a follow-up — if you add a provider, add the case that
routes to it.

If you changed a request, response, or FFI error type, regenerate the bindings
and commit the result. The Python development environment must already be
synced; CI runs the same command and fails on any difference:

```bash
cargo run -p warpllm-codegen
```

### Adding a provider or model

This section is about the roster warpllm **ships** — a public provider everyone
gets. If instead you want to reach a model of your own, on your own hardware,
that is not a contribution at all: write your own roster file and point warpllm
at it, as [the README](README.md#your-own-models) describes. `specs.yaml` is not
the place for somebody's localhost.

The common path (if the provider speaks a protocol warpllm already knows), in order:

1.  Add the entry to
    [`crates/warpllm/src/registry/specs.yaml`](crates/warpllm/src/registry/specs.yaml),
    following the rules in that file's header.
1.  Run `cargo test -p warpllm`. The registry has both load-time gates and
    lints, and this is where a bad entry gets caught.
1.  Add a test under `crates/warpllm/tests/providers/`, alongside the existing
    `openai` and `deepseek` cases.

## Community

### Communication Channels

* **To report bugs and feature requests**: [GitHub Issues](https://github.com/warpllm/warpllm/issues)
* **To chat with the warpllm team (questions, ideas, reports)**:
[Discord](https://discord.gg/payU7Vzuq5)
* **To discuss amongst the community**: [Reddit](https://www.reddit.com/r/warpllm/)

On [Discord](https://discord.gg/payU7Vzuq5), three channels carry most of the
work:

* **`#dev`** — feature ideas before they're issues, and how we should build the
  ones that are. A new protocol, a change to the registry's rules, anything
  where the approach is the hard part: raise it here first. A short thread often
  saves a long PR review.
* **`#support`** — you hit something that doesn't work and want help. Setup,
  usage, a call that fails in a way you can't explain. File a
  [bug report](https://github.com/warpllm/warpllm/issues/new/choose) once it
  looks like a defect rather than a snag.
* **`#intro`** — say hello. Tell us what you're building and what brought you
  here; it's how we know who to point at which issue.

### Getting Help

* Check existing documentation and examples
* Search closed issues for similar problems
* Ask in `#support` on Discord for quick questions

### Recognition

We value all contributions! Contributors are:

* Listed in release notes
* Mentioned in our README

## Questions?

If you have any questions, ask them away at any of these channels:

[![Discord](https://img.shields.io/badge/Discord-warpllm-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/payU7Vzuq5)
[![Reddit](https://img.shields.io/badge/Reddit-r%2Fwarpllm-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/r/warpllm/)
