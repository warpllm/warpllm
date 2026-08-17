#!/usr/bin/env bash
# Every suite CI runs, in one command.
#
# `cargo test --workspace` covers ONE of the three. The Node and Python suites
# exercise their own wrappers over the same core, and a change to the core can
# break them while cargo stays green — which is exactly how a 501 quietly became
# a 400 on both binding surfaces and reached CI.
#
# Both bindings are REBUILT rather than assumed current. This is the trap worth
# knowing about: `uv sync --locked` audits the environment without recompiling
# the extension module, so pytest will happily pass against a `.so` built from
# code that no longer exists. A green run against a stale artifact is worse than
# a red one.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> rust"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo "==> node"
(cd bindings/node && npm run build && npm test)

echo "==> python"
(cd bindings/python && uv run --with maturin maturin develop --uv && uv run pytest)

echo "==> generated types"
# Codegen is a writer, so drift shows up as changed files rather than an exit
# code. Compare the generated trees BEFORE and AFTER the run — comparing
# against git instead would flag every hand-written change in the working tree
# as codegen drift.
GENERATED=(
  bindings/node/src-ts/generated
  bindings/python/python/warpllm/_generated
)
before=$(find "${GENERATED[@]}" -type f -not -path '*/__pycache__/*' -exec shasum {} + | sort)
cargo run -p warpllm-codegen --features warpllm/codegen >/dev/null
after=$(find "${GENERATED[@]}" -type f -not -path '*/__pycache__/*' -exec shasum {} + | sort)
if [ "$before" != "$after" ]; then
  echo "generated binding types were stale; commit the regenerated files:" >&2
  diff <(echo "$before") <(echo "$after") >&2 || true
  exit 1
fi

echo "all suites passed"
