// What a caller writes when the chunks start arriving. The types can be
// field-perfect and still be unpleasant to consume, and only code that reads
// them like a user says which.
//
// The transcripts are the Rust suite's, read across the workspace on purpose:
// one recorded corpus, checked by every language that has to survive it. A
// capture added for a new provider tightens all three at once.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import type { CreateChatCompletionStreamResponse } from '../src-ts/generated/types.js'

const TRANSCRIPTS = resolve(
  __dirname,
  '../../../crates/warpllm/tests/protocol/openai_compat/chat_completions/fixtures/transcript',
)

function transcript(name: string): CreateChatCompletionStreamResponse[] {
  return readFileSync(resolve(TRANSCRIPTS, name), 'utf8')
    // SSE terminates a line with CRLF, and a Windows checkout stores the
    // fixture that way too — so splitting on `\n` alone leaves a `\r` on every
    // payload, and `[DONE]` stops matching the filter below. Rust's
    // `str::lines()` drops it for free, which is why only this reader broke.
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data: '))
    .map((line) => line.slice('data: '.length))
    .filter((payload) => payload !== '[DONE]')
    // The one cast a consumer makes, at the wire boundary and nowhere after.
    .map((payload) => JSON.parse(payload) as CreateChatCompletionStreamResponse)
}

describe('iterating a stream', () => {
  it('accumulates text without a cast or a non-null assertion', () => {
    let text = ''
    let finishReason: string | null = null
    let totalTokens: number | undefined

    for (const chunk of transcript('openai-text.sse')) {
      for (const choice of chunk.choices) {
        // Absent on a chunk that carries only a tool call, null on the one
        // that opens a refusal: `??` covers both because the type admits both.
        text += choice.delta.content ?? ''
        finishReason = choice.finish_reason ?? finishReason
      }
      // Null on every chunk but the last, so a truthiness check is wrong and
      // the type is what says so.
      if (chunk.usage != null) {
        totalTokens = chunk.usage.total_tokens
      }
    }

    expect(text).toBe('Hello there!')
    expect(finishReason).toBe('stop')
    expect(totalTokens).toBe(14)
  })

  it('joins tool-call fragments by index', () => {
    const argumentsByIndex = new Map<number, string>()
    const namesByIndex = new Map<number, string>()

    for (const chunk of transcript('deepseek-tool-call.sse')) {
      for (const call of chunk.choices.flatMap((choice) => choice.delta.tool_calls ?? [])) {
        // `index` is the only field every fragment carries; `id`, `type` and
        // the function's `name` arrive once, on whichever fragment opens the
        // call.
        argumentsByIndex.set(
          call.index,
          (argumentsByIndex.get(call.index) ?? '') + (call.function?.arguments ?? ''),
        )
        if (call.function?.name != null) {
          namesByIndex.set(call.index, call.function.name)
        }
      }
    }

    expect(namesByIndex.get(0)).toBe('get_weather')
    expect(JSON.parse(argumentsByIndex.get(0) ?? '')).toEqual({ city: 'Seoul' })
  })

  it('carries fields warpllm does not model through to the caller', () => {
    // `obfuscation` is on every live OpenAI chunk and in no specification.
    // Rust keeps it in `unknown_fields`, so it survives to here — untyped,
    // which is the deal the catch-all makes.
    const [first] = transcript('openai-text.sse')
    expect((first as Record<string, unknown>).obfuscation).toBe('KtQ3nZ8w')
  })
})
