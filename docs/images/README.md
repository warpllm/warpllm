# Site images

Empty on purpose. warpllm has no logo yet, so `docs.json` carries no `logo` and
no `favicon` field, and Mintlify renders the `name` value — `warpllm` — as a
text wordmark in the navbar. It styles that for light and dark itself, so there
is nothing to maintain until a real mark exists.

## Adding a logo

Drop the files in beside this README, then add the two fields to `docs.json`.
Nothing else in the site needs to change.

### 1. The files

| File | What it is | Notes |
| --- | --- | --- |
| `logo-light.svg` | Navbar logo on light backgrounds | Dark ink |
| `logo-dark.svg` | Navbar logo on dark backgrounds | Light ink |
| `favicon.svg` | Browser tab icon | Square, square-cropped artwork |

- **SVG is preferred** for the logo — it scales, and it stays crisp on a
  navbar that renders it about 26px tall. PNG works; make it at least 2x.
- **Two logo files, not one.** The navbar background flips with the theme, so a
  single-colour mark disappears in one of them. If your mark is theme-neutral,
  point both fields at the same file.
- **Transparent background** on both logo files.
- The favicon is **automatically resized** by Mintlify, so one square source is
  enough.

### 2. The config

Add these two keys to `docs.json`, next to `colors`:

```json
  "favicon": "/images/favicon.svg",
  "logo": {
    "light": "/images/logo-light.svg",
    "dark": "/images/logo-dark.svg",
    "href": "https://github.com/warpllm/warpllm"
  },
```

Paths are absolute from the content root — this directory is `docs/images/`, so
the path is `/images/…`, not `/docs/images/…`.

`href` is where clicking the logo goes. Drop the line to send it to the docs
home instead; drop the whole object and the text wordmark comes back.

### 3. Check it

```bash
cd docs && npx mint dev
```

Look at the navbar in both themes. `npx mint validate` catches a path that
points at a file that is not there.

## While you are picking a brand

`colors.primary` in `docs.json` is still `#7C5CFF`, which is a placeholder too —
it drives links, the active sidebar item, and the API playground's accent.
`colors.light` and `colors.dark` are the variants used on dark and light
backgrounds respectively. Set all three when the mark lands.
