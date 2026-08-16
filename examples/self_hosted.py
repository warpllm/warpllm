"""One chat completion against a model you run yourself.

Start a server that speaks the OpenAI API — vLLM, TGI, Ollama, llama.cpp —
then point warpllm at a roster describing it. No key, no fork:

    python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.3-70B-Instruct
    python examples/self_hosted.py

`warpllm.yaml` next door is the roster. It is MERGED over warpllm's built-in
one, so `openai/gpt-5-nano` still routes from this same client — the last call
below proves it, and needs OPENAI_API_KEY set.
"""

from pathlib import Path

from warpllm import WarpLLM

client = WarpLLM(specs_path=str(Path(__file__).parent / "warpllm.yaml"))

completion = client.chat_completions(
    {
        "model": "vllm/llama-3.3-70b",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
)

print(completion["choices"][0]["message"]["content"])

# The same client still reaches everything warpllm ships. A roster of your own
# adds to the list; it does not replace it.
print(client.chat_completions(
    {
        "model": "openai/gpt-5-nano",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
)["choices"][0]["message"]["content"])
