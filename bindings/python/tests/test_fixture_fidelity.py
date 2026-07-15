"""Hydrate every shared fixture through ChatCompletion.from_dict and compare
the result against the raw body. A field the API sends that from_dict drops
or renames fails the diff. (Absent-vs-null is normalized away: the Rust
round-trip test is the byte-exact gate; this guards the Python hydration.)"""

import dataclasses
import json
from pathlib import Path

import pytest
from warpllm import ChatCompletion

FIXTURES = sorted(
    (
        Path(__file__).resolve().parents[3]
        / "crates/warpllm/tests/types/openai/chat/completions/fixtures"
    ).glob("*.json")
)


def _prune_nones(value):
    if isinstance(value, dict):
        return {k: _prune_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune_nones(v) for v in value]
    return value


def test_fixtures_exist():
    assert FIXTURES, "no OpenAI fixtures found"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_hydrates_losslessly(path: Path):
    body = json.loads(path.read_text())
    completion = ChatCompletion.from_dict(body)
    assert _prune_nones(dataclasses.asdict(completion)) == _prune_nones(body)
