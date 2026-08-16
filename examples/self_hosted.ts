/**
 * One chat completion against a model you run yourself.
 *
 * Start a server that speaks the OpenAI API — vLLM, TGI, Ollama, llama.cpp —
 * then point warpllm at a roster describing it. No key, no fork:
 *
 *   ollama serve && ollama pull llama3.3
 *   node --experimental-strip-types examples/self_hosted.ts
 *
 * `warpllm.yaml` next door is the roster. It is MERGED over warpllm's built-in
 * one, so `openai/gpt-5-nano` still routes from this same client — the last
 * call below proves it, and needs OPENAI_API_KEY set.
 */

import { WarpLLM } from '@warpllm/warpllm'

const client = new WarpLLM({ specsPath: new URL('warpllm.yaml', import.meta.url).pathname })

const completion = await client.chatCompletions({
  model: 'ollama/llama3.3',
  messages: [{ role: 'user', content: 'Hello!' }],
})

console.log(completion.choices[0].message.content)

// The same client still reaches everything warpllm ships. A roster of your own
// adds to the list; it does not replace it.
const shipped = await client.chatCompletions({
  model: 'openai/gpt-5-nano',
  messages: [{ role: 'user', content: 'Hello!' }],
})

console.log(shipped.choices[0].message.content)
