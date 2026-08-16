# warpllm documentation

The Mintlify site. Everything published lives in this directory; `docs.json` is
the site config and the single source of navigation.

## Local preview

```bash
cd docs && npx mint dev
```

Serves on <http://localhost:3000> with hot reload. First boot takes ~80s while
the CLI installs and builds; after that, edits appear on save.

> **The CLI refuses to run on Node 25+.** Use an LTS release — `nvm install 24
> && nvm use 24`. If you would rather not add a Node version, run it in a
> container instead:
>
> ```bash
> docker run --rm -p 3000:3000 -v "$PWD":/work -w /work/docs node:24-slim \
>   npx -y mint dev
> ```

## Before you push

```bash
cd docs
npx mint validate       # docs.json schema and frontmatter
npx mint broken-links   # every internal link resolves
```

Both are quiet on success. `mint dev` also prints an error/warning count on
each build — keep it at zero.

## Deployment

The GitHub app builds on push to the default branch. Because `docs.json` is not
at the repository root, the Mintlify dashboard needs **Settings → Git settings →
Subdirectory** set to `docs`. Page slugs are then relative to this directory:
`docs/guides/streaming.mdx` publishes at `/guides/streaming`.

## Layout

```
docs/
├── docs.json                 site config — navigation lives here and nowhere else
├── index.mdx                 landing page (published at /)
├── quickstart.mdx
├── get-started/              install, authentication
├── concepts/                 model routing, requests, errors, timeouts
├── guides/                   streaming, tools, structured outputs, migration
├── models/                   the roster, and how to extend it
├── sdk/                      reference.mdx — all three languages, one page
├── gateway/                  the self-hosted HTTP server
├── api-reference/            OpenAPI spec + the pages that render it
├── snippets/                 reusable MDX, imported by pages
├── scripts/                  generators
└── images/                   logo and favicon — empty until a mark exists
```

### Where things go

| Adding | Put it in |
| --- | --- |
| A how-to with a task in the title | `guides/` |
| An explanation of how something works | `concepts/` |
| Language-specific API surface | a tab inside `sdk/reference.mdx` |
| Anything about running `warpllm-server` | `gateway/` |
| An HTTP endpoint | `api-reference/openapi.json` + a page |
| Text repeated on 2+ pages | `snippets/` |

## Branding

There is no logo yet. `docs.json` carries no `logo` and no `favicon` field, so
Mintlify renders the `name` value — `warpllm` — as a text wordmark in the
navbar, styled for light and dark automatically.

Adding a real mark is dropping three files into `docs/images/` and adding two
keys to `docs.json`. [`docs/images/README.md`](images/README.md) has the exact
filenames, the JSON to paste, and the sizing notes.

`colors.primary` is a placeholder too — see the same file.

## Adding a page

1. Create the `.mdx` file with frontmatter — `title` is required, `description`
   feeds SEO and search.
2. Add its path to `docs.json`. **A page not in `docs.json` does not appear in
   the sidebar.**
3. Link to it from a related page so it is reachable by reading, not only by
   searching.

```mdx
---
title: "Page title in sentence case"
description: "One sentence, for SEO and search results."
icon: "book-open"
---
```

## Generated content — do not edit by hand

Two snippets are generated from the model registry
(`crates/warpllm/src/registry/specs.yaml`):

- `snippets/model-roster.mdx` — every provider and model with its limits
- `snippets/provider-keys.mdx` — provider → environment variable

Regenerate after any registry change, in the same commit:

```bash
uv run --with pyyaml docs/scripts/gen_model_roster.py
```

`--check` writes nothing and exits 1 if either snippet is stale. Worth wiring
into CI so a registry change cannot ship without the docs following:

```yaml
- name: Docs match the model registry
  run: uv run --with pyyaml docs/scripts/gen_model_roster.py --check
```

## Conventions

- **Filenames** are kebab-case: `migrate-from-openai.mdx`.
- **Internal links** are root-relative with no extension: `/guides/streaming`.
- **Headings** are sentence case.
- **Code blocks** always carry a language tag. In a `CodeGroup`, the text after
  the language is the tab label: ` ```python Python `.
- **Multi-language examples** go in a `CodeGroup` or `Tabs` block, always
  labelled exactly `Python`, `TypeScript`, `Rust`, in that order. The labels
  are load-bearing: Mintlify syncs tabs with matching titles across a page, so
  a reader who picks `Rust` in one block gets Rust in all of them. A label
  typo silently breaks that.
- **Voice** is second person and active. No marketing adjectives.
- **Snippets** import from the content root: `import X from '/snippets/x.mdx'`.
  Component names must start with a capital.

## The SDK reference is one page

`sdk/reference.mdx` covers Python, TypeScript, and Rust in a single file, with
each section wrapped in a `<Tabs>` block. Mintlify keeps tabs with matching
titles in sync across a page, so the first block acts as a language selector
for the whole reference. The selection does not persist across pages or
reloads — it is per-page state, which is exactly the scope this needs.

Adding a section means adding one `<Tabs>` with all three `<Tab>`s. If a
feature genuinely has no equivalent in one language, say so inside that
language's tab rather than dropping the tab — a missing tab makes the sync
jump to a different language.

## Accuracy rules

These docs describe an early project, and the difference between shipped and
unshipped is load-bearing.

- Anything about the HTTP gateway carries the `gateway-preview` snippet and a
  `tag: "Preview"` in its frontmatter. The gateway is on `main` and unreleased.
- Version-specific claims name the version (`0.4.0`).
- A capability warpllm does not have is stated as absent rather than omitted.
  Say **not yet** for work that is planned or tracked, and link the tracking
  issue. Do not call a gap "deliberate" — it reads as "never", and on a
  pre-1.0 project it will be wrong.
- Code samples are taken from the real API surface. `examples/` in the
  repository root holds compiled, runnable versions of the quickstarts; prefer
  copying from there over writing fresh Rust.
