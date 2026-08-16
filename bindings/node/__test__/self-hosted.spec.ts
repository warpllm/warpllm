import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, expect, test } from 'vitest'

import { InternalServerError, WarpLLM } from '../dist/index.js'
import { MockServer } from './mock-server.js'

/**
 * `specsPath` crosses the FFI boundary and routes a self-hosted model.
 *
 * Deliberately thin. What a roster file MEANS — the merge over the built-in
 * roster, the `Authorization` header that must NOT be sent, which checks a
 * stranger's file is held to — is proved in Rust, over the same code these
 * call into. What only this side can prove is that the option reaches it at
 * all: the config crosses as an opaque JSON string, so a misspelled key would
 * be dropped here and rejected there.
 */

let server: MockServer

beforeEach(async () => {
  server = await MockServer.start()
})

afterEach(async () => {
  await server.close()
})

/** A roster naming one self-hosted provider that takes no credential. */
const roster = (baseUrl: string): string => {
  const path = join(mkdtempSync(join(tmpdir(), 'warpllm-')), 'warpllm.yaml')
  writeFileSync(
    path,
    [
      'providers:',
      '  local:',
      `    base_url: "${baseUrl}"`,
      '    auth: none',
      '    models:',
      '      local/llama-3.3-70b:',
      '        supported_apis:',
      '          - {api: openai_compat_chat_completions}',
      '',
    ].join('\n'),
  )
  return path
}

test('specsPath routes to a self-hosted model with no credential', async () => {
  server.respondWith(200, {
    id: 'chatcmpl-local',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: 'llama-3.3-70b',
    choices: [
      { index: 0, message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' },
    ],
  })

  const client = new WarpLLM({ specsPath: roster(server.url), timeout: 5 })
  const response = await client.chatCompletions({
    model: 'local/llama-3.3-70b',
    messages: [{ role: 'user', content: 'hi' }],
  })

  expect(response.model).toBe('local/llama-3.3-70b')
  expect(server.requests[0].headers.authorization).toBeUndefined()
})

test('a bad roster throws at construction, not on the first request', () => {
  const path = join(mkdtempSync(join(tmpdir(), 'warpllm-')), 'warpllm.yaml')
  writeFileSync(path, 'providers:\n  local:\n    base_url_typo: x\n')
  expect(() => new WarpLLM({ specsPath: path })).toThrow(InternalServerError)
})

test('a missing roster throws at construction', () => {
  expect(() => new WarpLLM({ specsPath: './no-such-roster.yaml' })).toThrow(
    InternalServerError,
  )
})
