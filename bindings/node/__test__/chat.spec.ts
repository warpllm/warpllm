import { afterEach, beforeEach, expect, test } from 'vitest'

import {
  AuthenticationError,
  InvalidRequestError,
  NotImplementedError,
  WarpLLM,
} from '../dist/index.js'
import { MockServer } from './mock-server.js'

const MESSAGES = [{ role: 'user' as const, content: 'hi' }]

const OPENAI_COMPLETION = {
  id: 'chatcmpl-123',
  object: 'chat.completion',
  created: 1_700_000_000,
  model: 'gpt-4o-2024-08-06',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Hello there!' },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 9,
    completion_tokens: 12,
    total_tokens: 21,
    prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 2, audio_tokens: 0 },
    completion_tokens_details: {
      reasoning_tokens: 5,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
  },
  service_tier: 'default',
  system_fingerprint: 'fp_44709d6fcb',
}

let server: MockServer
let client: WarpLLM

beforeEach(async () => {
  server = await MockServer.start()
  client = new WarpLLM({
    openaiApiKey: 'sk-test-openai',
    baseUrl: server.url,
    timeout: 5,
  })
})

afterEach(async () => {
  await server.close()
})

test('openai happy path', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  const completion = await client.chat.completions.create({
    model: 'openai/gpt-4o',
    messages: MESSAGES,
  })

  expect(completion.choices[0].message.content).toBe('Hello there!')
  expect(completion.choices[0].finish_reason).toBe('stop')
  expect(completion.model).toBe('openai/gpt-4o')
  expect(completion.usage?.total_tokens).toBe(21)
  expect(completion.service_tier).toBe('default')
  expect(completion.system_fingerprint).toBe('fp_44709d6fcb')
  expect(completion.usage?.prompt_tokens_details?.cached_tokens).toBe(3)
  expect(completion.usage?.prompt_tokens_details?.cache_write_tokens).toBe(2)
  expect(completion.usage?.completion_tokens_details?.reasoning_tokens).toBe(5)

  const sent = server.requests[0]
  expect(sent.url).toBe('/chat/completions')
  expect(sent.headers.authorization).toBe('Bearer sk-test-openai')
  // Provider prefix stripped from the outbound model.
  expect((sent.body as { model: string }).model).toBe('gpt-4o')
})

test('401 rejects with AuthenticationError', async () => {
  server.respondWith(401, {
    error: { message: 'Incorrect API key provided', type: 'invalid_request_error' },
  })

  const err = await client.chat.completions
    .create({ model: 'openai/gpt-4o', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(AuthenticationError)
  expect((err as AuthenticationError).status).toBe(401)
  expect((err as AuthenticationError).provider).toBe('openai')
  expect((err as AuthenticationError).message).toContain('Incorrect API key')
})

test('invalid model rejects unsupported provider', async () => {
  const err = await client.chat.completions
    .create({ model: 'mistral/large', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(InvalidRequestError)
  expect((err as InvalidRequestError).message).toContain('not a supported provider')
})

test('bare model name is rejected', async () => {
  const err = await client.chat.completions
    .create({ model: 'gpt-4o', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(InvalidRequestError)
  expect((err as InvalidRequestError).message).toContain('not a supported provider')
})

test('missing baseUrl rejects without a provider request', async () => {
  client = new WarpLLM({ openaiApiKey: 'sk-test-openai' })

  const err = await client.chat.completions
    .create({ model: 'openai/gpt-4o', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(InvalidRequestError)
  expect((err as InvalidRequestError).message).toContain('missing base_url')
  expect(server.requests).toHaveLength(0)
})

test('stream: true rejects as not implemented', async () => {
  const err = await client.chat.completions
    .create({ model: 'openai/gpt-4o', messages: MESSAGES, stream: true })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(NotImplementedError)
  expect(server.requests).toHaveLength(0)
})
