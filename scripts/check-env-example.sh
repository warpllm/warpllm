#!/usr/bin/env bash
#
# `examples/.env.example` calls itself a mirror of the roster. This is what
# makes that true.
#
# Every `env_api_key:` in `specs.yaml` names a variable somebody has to export,
# and the template is where they find out which. Nothing else checks the two
# against each other: the roster's own tests read `specs.yaml` alone, and this
# cannot become one of them — `.env.example` sits outside the crate directory,
# so `include_str!` of it would break `cargo package`.
#
# Which is exactly how `MOONSHOT_API_KEY` went missing for a whole release: the
# `kimi` provider landed, the template did not, and the file's own warning that
# no test guarded it turned out to be right.
#
# A mirror has two faces, so this checks both. A variable in the roster and not
# in the template is the `kimi` case above. A variable in the template and not
# in the roster is the opposite mistake — a provider renamed or dropped, its old
# assignment left behind — and it is worse than untidy: the template is what a
# reader trusts to say which keys warpllm actually reads, so a stale line sends
# somebody hunting for a key nothing will ever look at.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Overridable so `check-env-example.test.sh` can run the real logic against a
# pair of fixture files. Nothing else sets them; CI and `test-all.sh` take the
# defaults.
specs="${CHECK_ENV_SPECS:-$root/crates/warpllm/src/registry/specs.yaml}"
template="${CHECK_ENV_TEMPLATE:-$root/examples/.env.example}"

# Variables the template sets that are deliberately NOT a provider's
# `env_api_key`, and so have no counterpart in the roster to match against.
#
# A name earns a place here only if warpllm reads it from the environment for
# something other than authenticating a provider. `WARPLLM_SPECS` is a path to a
# roster file, which is the whole of that category today. A provider's key never
# belongs here — that is precisely the drift this script exists to catch.
#
# The list lives in the script rather than as a comment in the template because
# skipping it is what breaks the check, and prose cannot be skipped loudly.
non_provider=(WARPLLM_SPECS)

# `$2` of `env_api_key: FOO`, which the roster schema requires to be a bare
# scalar.
roster_vars="$(grep -oE '^[[:space:]]*env_api_key:[[:space:]]*[^[:space:]]+' "$specs" |
  awk '{print $2}' | sort -u)"

# `^VAR=` and not a bare match: the variable has to be a SETTABLE line, not
# merely mentioned in a comment above one.
template_vars="$(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$template" | tr -d '=' | sort -u)"
template_vars="$(comm -23 \
  <(printf '%s\n' "$template_vars") \
  <(printf '%s\n' "${non_provider[@]}" | sort -u))"

missing="$(comm -23 <(printf '%s\n' "$roster_vars") <(printf '%s\n' "$template_vars"))"
stale="$(comm -13 <(printf '%s\n' "$roster_vars") <(printf '%s\n' "$template_vars"))"

# Reported apart, because the remedies are opposite: one wants a block written,
# the other wants a block deleted. Collapsing them into one "these don't match"
# list would leave the reader to work out which way round it went.
if [ -n "$missing" ] || [ -n "$stale" ]; then
  echo "examples/.env.example is out of step with the roster." >&2
  if [ -n "$missing" ]; then
    echo >&2
    echo "In specs.yaml, missing from the template:" >&2
    printf '  %s\n' $missing >&2
    echo "  -> add a block per variable, following the ones already there." >&2
  fi
  if [ -n "$stale" ]; then
    echo >&2
    echo "In the template, no longer an env_api_key in specs.yaml:" >&2
    printf '  %s\n' $stale >&2
    echo "  -> delete the block, or add the name to non_provider in this script" >&2
    echo "     if warpllm really does read it for something else." >&2
  fi
  exit 1
fi

echo "examples/.env.example mirrors every env_api_key in specs.yaml"
