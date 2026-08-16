#!/usr/bin/env bash
#
# `check-env-example.sh` is the only thing holding `examples/.env.example` to
# the roster, so a bug in it is silent by construction — the check keeps passing
# and the drift it was written to catch walks straight through.
#
# These cases drive the real script against fixture pairs, one per way the two
# files can disagree, plus the allowlist entry that is in the template on
# purpose and must not read as drift.
set -euo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check-env-example.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0

# Runs the check over one fixture pair. `expect` is `pass` or `fail`; the
# remaining arguments are text the output has to contain, which is what keeps a
# case honest — a script that failed for an unrelated reason would otherwise
# read as a pass.
run_case() {
  local name="$1" expect="$2" specs_body="$3" template_body="$4"
  shift 4

  printf '%s' "$specs_body" >"$work/specs.yaml"
  printf '%s' "$template_body" >"$work/.env.example"

  local output status=0
  output="$(CHECK_ENV_SPECS="$work/specs.yaml" CHECK_ENV_TEMPLATE="$work/.env.example" \
    "$script" 2>&1)" || status=$?

  if [ "$expect" = pass ] && [ "$status" -ne 0 ]; then
    echo "FAIL $name: expected the check to pass, it exited $status" >&2
    printf '%s\n' "$output" >&2
    failures=$((failures + 1))
    return
  fi
  if [ "$expect" = fail ] && [ "$status" -eq 0 ]; then
    echo "FAIL $name: expected the check to fail, it passed" >&2
    printf '%s\n' "$output" >&2
    failures=$((failures + 1))
    return
  fi

  local needle
  for needle in "$@"; do
    if ! printf '%s' "$output" | grep -qF -- "$needle"; then
      echo "FAIL $name: output did not mention '$needle'" >&2
      printf '%s\n' "$output" >&2
      failures=$((failures + 1))
      return
    fi
  done

  echo "ok   $name"
}

ROSTER_ONE='providers:
  alpha:
    base_url: "https://alpha.example/v1"
    env_api_key: ALPHA_API_KEY
'
ROSTER_TWO="$ROSTER_ONE"'  beta:
    base_url: "https://beta.example/v1"
    env_api_key: BETA_API_KEY
'
TEMPLATE_ONE='ALPHA_API_KEY=
'

run_case "a template that mirrors the roster passes" pass \
  "$ROSTER_ONE" "$TEMPLATE_ONE" \
  "mirrors every env_api_key"

run_case "a roster variable with no block is caught" fail \
  "$ROSTER_TWO" "$TEMPLATE_ONE" \
  "missing from the template" "BETA_API_KEY"

run_case "a template block with no roster entry is caught" fail \
  "$ROSTER_ONE" "$TEMPLATE_ONE"'GAMMA_API_KEY=
' \
  "no longer an env_api_key" "GAMMA_API_KEY"

# The case a naive set-equality check gets wrong on day one, and the reason the
# allowlist exists at all.
run_case "the allowlisted WARPLLM_SPECS is not drift" pass \
  "$ROSTER_ONE" "$TEMPLATE_ONE"'WARPLLM_SPECS=
' \
  "mirrors every env_api_key"

# Both directions at once report both, rather than the first one found.
run_case "the two directions are reported apart" fail \
  "$ROSTER_TWO" 'GAMMA_API_KEY=
ALPHA_API_KEY=
' \
  "missing from the template" "BETA_API_KEY" \
  "no longer an env_api_key" "GAMMA_API_KEY"

# A variable named only in a comment is not set, and the reader who copies this
# file gets nothing from it.
run_case "a variable mentioned only in a comment does not count" fail \
  "$ROSTER_TWO" "$TEMPLATE_ONE"'# BETA_API_KEY= is what you would export
' \
  "missing from the template" "BETA_API_KEY"

if [ "$failures" -gt 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi

echo "check-env-example.sh: all cases passed"
