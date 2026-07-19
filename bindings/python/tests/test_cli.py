"""CLI surface tests mirroring the Node binding's cli.spec.ts."""

import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "warpllm._cli", *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_help_lists_every_flag_and_exits_zero():
    result = run_cli("--help")
    assert result.returncode == 0
    for expected in ("--host", "--port", "--timeout-secs"):
        assert expected in result.stdout


def test_unknown_flag_surfaces_shared_parser_error():
    result = run_cli("--bogus")
    assert result.returncode == 1
    assert "unexpected argument" in result.stderr
    assert "--bogus" in result.stderr


def boot_gateway(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "warpllm._cli",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_health(port: int) -> None:
    for _ in range(50):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as health:
                assert health.status == 200
                assert json.load(health)["status"] == "ok"
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("gateway never came up")


def test_cli_boots_gateway_and_answers_health():
    port = free_port()
    child = boot_gateway(port)
    try:
        wait_for_health(port)
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform == "win32", reason="no SIGINT delivery on Windows"
)
def test_sigint_shuts_down_gracefully_and_exits_zero():
    port = free_port()
    child = boot_gateway(port)
    try:
        wait_for_health(port)
        child.send_signal(signal.SIGINT)
        assert child.wait(timeout=10) == 0
    finally:
        child.kill()
        child.wait(timeout=5)
