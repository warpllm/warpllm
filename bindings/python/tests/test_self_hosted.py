"""`specs_path` crosses the FFI boundary and routes a self-hosted model.

Deliberately thin. What a roster file MEANS -- the merge, the missing
`Authorization` header, which checks a stranger's file is held to -- is proved
in Rust, over the same code these call into. What can only be proved from this
side is that the keyword reaches it at all, since the config crosses as an
opaque JSON string and a misspelled key would be silently dropped by neither
language and loudly rejected by Rust.
"""

from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer
from warpllm import InternalServerError, WarpLLM


def _roster(tmp_path: Path, base_url: str) -> str:
    """A roster naming one self-hosted provider that takes no credential.

    The trailing slash is stripped because a roster with one does not load --
    endpoints append their own path, and warpllm refuses the doubled slash at
    construction rather than putting it on the wire. The `base_url` fixture
    hands back `http://host:port/`, so this is the same trim a real reader
    makes when copying an address out of their server's logs.
    """
    path = tmp_path / "warpllm.yaml"
    path.write_text(
        "providers:\n"
        "  local:\n"
        f'    base_url: "{base_url.rstrip("/")}"\n'
        "    auth: none\n"
        "    models:\n"
        "      local/llama-3.3-70b:\n"
        "        supported_apis:\n"
        "          - {api: openai_compat_chat_completions}\n"
    )
    return str(path)


def test_specs_path_routes_to_a_self_hosted_model(
    tmp_path: Path, httpserver: HTTPServer, base_url: str
) -> None:
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "llama-3.3-70b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    client = WarpLLM(specs_path=_roster(tmp_path, base_url), timeout=5)
    response = client.chat_completions(
        {
            "model": "local/llama-3.3-70b",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert response["model"] == "local/llama-3.3-70b"
    # No key was set for `local`, and none was needed.
    assert "Authorization" not in httpserver.log[0][0].headers


def test_a_bad_roster_raises_at_construction(tmp_path: Path) -> None:
    """The file is read when the client is built, so the failure lands where
    the caller is holding the path -- not on a request much later."""
    path = tmp_path / "warpllm.yaml"
    path.write_text("providers:\n  local:\n    base_url_typo: x\n")
    with pytest.raises(InternalServerError) as raised:
        WarpLLM(specs_path=str(path))
    assert "unknown field" in str(raised.value)


def test_a_missing_roster_raises_at_construction(tmp_path: Path) -> None:
    with pytest.raises(InternalServerError):
        WarpLLM(specs_path=str(tmp_path / "not-a-file.yaml"))
