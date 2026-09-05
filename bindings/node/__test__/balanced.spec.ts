import { afterEach, beforeEach, expect, test } from 'vitest'

import { BadRequestError, WarpLLMBalanced, type BalancedCandidate } from '../dist/index.js'
import { MockServer } from './mock-server.js'

const MESSAGES = [{ role: 'user', content: 'hi' }]

const completion = (model: string) => ({
  id: 'chatcmpl-1',
  object: 'chat.completion',
  created: 1_700_000_000,
  model,
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'hi' },
      finish_reason: 'stop',
    },
  ],
})

let server: MockServer

beforeEach(async () => {
  server = await MockServer.start()
  process.env.OPENAI_API_KEY = 'sk-test-openai'
  process.env.DEEPSEEK_API_KEY = 'sk-test-deepseek'
})

afterEach(async () => {
  await server.close()
})

test('all-zero weight is rejected', () => {
  expect(
    () =>
      new WarpLLMBalanced([{ model: 'openai/gpt-5.6', weight: 0 }], {
        baseUrl: server.url,
      }),
  ).toThrow(BadRequestError)
})

test('a weight exceeding i32::MAX is rejected', () => {
  expect(
    () =>
      new WarpLLMBalanced([{ model: 'openai/gpt-5.6', weight: 4_294_967_295 }], {
        baseUrl: server.url,
      }),
  ).toThrow(/exceeds the maximum/)
})

test('a candidate missing weight raises before reaching Rust', () => {
  // `as` bypasses the compile-time check on purpose -- this is the shape a
  // plain-JS caller (no compiler at all) can actually send.
  const candidates = [{ model: 'openai/gpt-5.6' }] as BalancedCandidate[]
  expect(() => new WarpLLMBalanced(candidates, { baseUrl: server.url })).toThrow(
    /candidates\[0\]/,
  )
})

test('an unknown candidate model is rejected', () => {
  expect(
    () =>
      new WarpLLMBalanced([{ model: 'openai/not-a-model', weight: 1 }], {
        baseUrl: server.url,
      }),
  ).toThrow(/no registered model/)
})

test("the request's model is rewritten to the selected candidate", async () => {
  server.respondWith(200, completion('gpt-5.6'))
  const client = new WarpLLMBalanced([{ model: 'openai/gpt-5.6', weight: 1 }], {
    baseUrl: server.url,
    timeout: 5,
  })
  const result = await client.chatCompletions({ model: 'my-group', messages: MESSAGES })
  expect(result.model).toBe('openai/gpt-5.6')
  expect((server.requests[0].body as { model: string }).model).toBe('gpt-5.6')
})

test('weighted distribution across two candidates', async () => {
  for (let i = 0; i < 4; i++) server.respondWith(200, completion('ignored'))
  const client = new WarpLLMBalanced(
    [
      { model: 'openai/gpt-5.6', weight: 3 },
      { model: 'deepseek/deepseek-v4-flash', weight: 1 },
    ],
    { baseUrl: server.url, timeout: 5 },
  )
  for (let i = 0; i < 4; i++) {
    await client.chatCompletions({ model: 'my-group', messages: MESSAGES })
  }
  const sentModels = server.requests.map((r) => (r.body as { model: string }).model)
  // Exact distribution over one full cycle of weights [3, 1].
  expect(sentModels.filter((m) => m === 'gpt-5.6')).toHaveLength(3)
  expect(sentModels.filter((m) => m === 'deepseek-v4-flash')).toHaveLength(1)
})
