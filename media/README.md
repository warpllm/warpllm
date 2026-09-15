# README assets

Images the root `README.md` shows. Anything else — the docs site's own
artwork — belongs in [`docs/images/`](../docs/images/README.md) instead.

Reference these files by their **absolute** `raw.githubusercontent.com` URL,
never by a relative path. The root README is published verbatim as the README
of all three packages (`readme = "../../README.md"` for the crate and the
Python wheel, a `prepack` copy for npm), and PyPI renders a relative image path
as a broken image.

| File | What it is | Source |
| --- | --- | --- |
| `tempest-wordmark.png` | Tempest's wordmark, shown under Partners | [tempestai-dev/tempest](https://github.com/tempestai-dev/tempest/blob/main/media/wordmark.png), used with their agreement |

Third-party marks are committed here rather than hotlinked from the owner's
repo: that URL tracks their default branch and breaks silently if the file
moves. The trade is that a rebrand needs a PR here, which is the right way
round for someone else's logo.
