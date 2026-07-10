import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { echo, version } from '../index.js'

// vitest runs with cwd = bindings/node (npm scripts run from the package dir)
const workspaceVersion = readFileSync(join(process.cwd(), '../../Cargo.toml'), 'utf8')
  .match(/\[workspace\.package\][^[]*?version\s*=\s*"([^"]+)"/)![1]

test('version matches the workspace Cargo.toml (single source of truth)', () => {
  expect(version()).toBe(workspaceVersion)
})

test('package.json version matches the workspace Cargo.toml', () => {
  const pkg = JSON.parse(readFileSync(join(process.cwd(), 'package.json'), 'utf8'))
  expect(pkg.version).toBe(workspaceVersion)
})

test('echo round-trips through a Promise', async () => {
  await expect(echo('hi')).resolves.toBe('hi')
})

test('echo rejects empty input', async () => {
  await expect(echo('')).rejects.toThrow('message must not be empty')
})
