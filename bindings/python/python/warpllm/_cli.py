"""`uvx warpllm` / `warpllm` entry point: boots the OpenAI-compatible gateway.

Flags go verbatim to the shared Rust parser -- run with --help for options.
Ctrl+C is handled by the native layer (tokio), which shuts the server down
gracefully; the chained Python signal handler then re-raises it here as
KeyboardInterrupt, which we swallow to exit 0 like the Rust binary.
"""

import sys

from warpllm._warpllm import serve


def main() -> None:
    try:
        # --help prints and returns; a running server blocks
        serve(sys.argv[1:])
    except KeyboardInterrupt:
        pass  # the server already shut down gracefully
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
