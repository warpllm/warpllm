import { afterEach, beforeEach, expect, test } from 'vitest'

import {
  APIError,
  AuthenticationError,
  BadRequestError,
  InternalServerError,
  RateLimitError,
  WarpLLM,
} from '../dist/index.js'
import { MockServer } from './mock-server.js'

const MESSAGES = [{ role: 'user', content: 'hi' }]

const request = (model = 'openai/gpt-5.6', extra: Record<string, unknown> = {}) => ({
  model,
  messages: MESSAGES,
  ...extra,
})

const OPENAI_COMPLETION = {
  id: 'chatcmpl-123',
  object: 'chat.completion',
  created: 1_700_000_000,
  model: 'gpt-5.6-2024-08-06',
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
  // The native client reads OPENAI_API_KEY at construction, so set it first.
  process.env.OPENAI_API_KEY = 'sk-test-openai'
  client = new WarpLLM({ baseUrl: server.url, timeout: 5 })
})

afterEach(async () => {
  await server.close()
})

const failure = async (req: Parameters<WarpLLM['chatCompletions']>[0]) => {
  const err = await client.chatCompletions(req).catch((e: unknown) => e)
  expect(err).toBeInstanceOf(APIError)
  return err as APIError
}

test('openai happy path', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  const completion = await client.chatCompletions(request())

  expect(completion.choices[0].message.content).toBe('Hello there!')
  expect(completion.choices[0].finish_reason).toBe('stop')
  expect(completion.model).toBe('openai/gpt-5.6')
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
  expect((sent.body as { model: string }).model).toBe('gpt-5.6')
})

// The request is forwarded verbatim rather than rebuilt field by field, so a
// parameter the wrapper does not model still reaches the provider. The old
// wrapper copied a fixed list of keys and silently dropped the rest.
//
// Written as a bare literal, not `{ ...request(), seed: 7 }`: spreading a
// `Record<string, unknown>` gives the literal an index signature of its own,
// which turns excess-property checking off and would let this pass however
// the declaration was generated. A fresh literal is the only form that
// actually tests it.
test('a request field the wrapper does not model still goes upstream', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  // No cast: an unmodelled OpenAI parameter has to type-check, or the
  // wrapper does not accept an OpenAI-compatible request.
  await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: MESSAGES,
    max_tokens: 64,
    seed: 7,
    response_format: { type: 'json_object' },
  })

  expect(server.requests[0].body).toMatchObject({
    max_tokens: 64,
    seed: 7,
    response_format: { type: 'json_object' },
  })
})

// The other half of that bargain, and the reason the generated declarations
// carry no index signature: an open request type would also open the response,
// and a misspelled field access would quietly type as `unknown` instead of
// failing the build. Checked at compile time — `@ts-expect-error` fails the
// typecheck if the line it guards ever stops being an error.
test('a misspelled response field does not compile', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  const completion = await client.chatCompletions(request())

  // @ts-expect-error `choicez` is not a field on CreateChatCompletionResponse
  expect(completion.choicez).toBeUndefined()
  expect(completion.choices).toHaveLength(1)
})

test('401 reports authentication', async () => {
  server.respondWith(401, {
    error: {
      message: 'Incorrect API key provided',
      type: 'invalid_request_error',
      code: 'invalid_api_key',
    },
  })

  const err = await failure(request())

  expect(err).toBeInstanceOf(AuthenticationError)
  expect(err.status).toBe(401)
  expect(err.message).toContain('Incorrect API key')
  expect(err.code).toBe('invalid_api_key')
  expect(err.type).toBe('invalid_request_error')
})

// A quota exhaustion arrives as a 429 and reads exactly like a rate limit,
// but no amount of backing off buys credit. A retry loop keyed on
// `code === 'rate_limited'` must not fire here — that is how a billing
// failure becomes an infinite retry loop.
test('quota exhaustion is not reported as a rate limit', async () => {
  server.respondWith(429, {
    error: {
      message: 'You exceeded your current quota',
      type: 'invalid_request_error',
      code: 'insufficient_quota',
    },
  })

  const err = await failure(request())

  // OpenAI reports both under one class, so the class cannot tell them
  // apart and `code` is the only thing that can.
  expect(err).toBeInstanceOf(RateLimitError)
  expect(err.status).toBe(429)
  expect(err.code).toBe('insufficient_quota')
  expect(err.code).not.toBe('rate_limit_exceeded')
})

test('a rate limit carries the provider’s request id', async () => {
  server.respondWith(
    429,
    { error: { message: 'Rate limit reached', type: 'rate_limit_error' } },
    { 'retry-after': '30', 'x-request-id': 'req-abc' },
  )

  const err = await failure(request())

  expect(err).toBeInstanceOf(RateLimitError)
  expect(err.type).toBe('rate_limit_error')
  // Both live only in headers, so they prove the transport kept them.
  expect(err.requestID).toBe('req-abc')
  expect(err.headers?.['retry-after']).toBe('30')
})

// A context overflow must not read as a plain bad request: the remedy is a
// shorter prompt or a bigger model, not a corrected payload.
test('context overflow is classified', async () => {
  server.respondWith(400, {
    error: {
      message: 'maximum context length is 8192 tokens',
      type: 'invalid_request_error',
      code: 'context_length_exceeded',
    },
  })

  const err = await failure(request())

  expect(err).toBeInstanceOf(BadRequestError)
  expect(err.code).toBe('context_length_exceeded')
})

// The two halves of one flat code space. A provider rejecting the request
// and warpllm rejecting it read almost alike, and the remedy is not the
// same — one edits the payload, the other may just need a different model.
test('code separates the provider’s rejection from warpllm’s', async () => {
  server.respondWith(400, {
    error: { message: 'bad payload', type: 'invalid_request_error' },
  })

  const upstream = await failure(request())
  // ...and warpllm's own rejection never left the process.
  const local = await failure(request('mistral/large'))

  expect(upstream).toBeInstanceOf(BadRequestError)
  expect(local).toBeInstanceOf(BadRequestError)
  expect(upstream.type).toBe('invalid_request_error')
  expect(local.type).toBe('invalid_request_error')
  // The provider named no code, and warpllm does not invent one for it.
  expect(upstream.code).toBeNull()
  expect(local.code).toBe('invalid_request')
})

test('invalid model rejects unsupported provider', async () => {
  expect((await failure(request('mistral/large'))).message).toContain(
    'no registered model spec',
  )
})

test('bare model name is rejected', async () => {
  expect((await failure(request('gpt-5.6'))).message).toContain('no registered model spec')
})

// The chunks a provider sends for "Hello there!", mirroring
// fixtures/transcript/openai-text.sse. Written out rather than read from the
// shared corpus because these assert the WRAPPER, not the shapes — the corpus
// is what stream-consumer.spec.ts checks.
const ENVELOPE =
  '"id":"chatcmpl-1","object":"chat.completion.chunk",' +
  '"created":1700000000,"model":"gpt-5.6"'

const OPENAI_STREAM =
  `data: {${ENVELOPE},"choices":[{"index":0,"delta":` +
  '{"role":"assistant","content":"Hello"},"logprobs":null,' +
  '"finish_reason":null}],"usage":null,"obfuscation":"KtQ3nZ8w"}\n\n' +
  ': keepalive\n\n' +
  `data: {${ENVELOPE},"choices":[{"index":0,"delta":` +
  '{"content":" there!"},"logprobs":null,"finish_reason":"stop"}]}\n\n' +
  'data: [DONE]\n\n'

test('a stream is iterated with for await', async () => {
  server.respondWithStream(OPENAI_STREAM)

  // `stream: true` inline, so the overload resolves to the streaming
  // signature; `request()` widens its extras to `unknown` and would not.
  const stream = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: MESSAGES,
    stream: true,
  })

  let text = ''
  const chunks = []
  for await (const chunk of stream) {
    chunks.push(chunk)
    // No cast and no non-null assertion: the generated types have to be
    // pleasant to read, not merely correct.
    text += chunk.choices[0]?.delta.content ?? ''
  }

  expect(text).toBe('Hello there!')
  // The sentinel is the stream ending, never a chunk.
  expect(chunks).toHaveLength(2)
  // Every chunk echoes the caller's prefixed string, not the upstream name.
  expect(chunks.every((c) => c.model === 'openai/gpt-5.6')).toBe(true)
  expect(chunks[1]?.choices[0]?.finish_reason).toBe('stop')
  // The request actually asked the provider to stream.
  expect((server.requests[0]?.body as { stream?: unknown }).stream).toBe(true)
})

// The Rust config sets `deny_unknown_fields`, so a key misspelled on the way
// across fails at CONSTRUCTION rather than being quietly ignored — which makes
// this a real check that the option arrives, not just that it is accepted here.
test('streamReadTimeout reaches the native config', async () => {
  server.respondWithStream(OPENAI_STREAM)
  const bounded = new WarpLLM({ baseUrl: server.url, timeout: 5, streamReadTimeout: 30 })

  const chunks = []
  for await (const chunk of await bounded.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: MESSAGES,
    stream: true,
  })) {
    chunks.push(chunk)
  }

  expect(chunks).toHaveLength(2)
})

test('a refusal before the stream opens raises the typed error', async () => {
  server.respondWith(429, {
    error: { message: 'Rate limit reached', type: 'rate_limit_exceeded' },
  })

  await expect(
    client.chatCompletions({
      model: 'openai/gpt-5.6',
      messages: MESSAGES,
      stream: true,
    }),
  ).rejects.toBeInstanceOf(RateLimitError)
})

test('an undecodable event ends the stream as a typed error', async () => {
  server.respondWithStream('data: not json\n\n')

  const stream = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: MESSAGES,
    stream: true,
  })

  await expect(async () => {
    for await (const _chunk of stream) {
      // Reaching here at all would mean a malformed event became a chunk.
    }
  }).rejects.toBeInstanceOf(APIError)
})
