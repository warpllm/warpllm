import { spawn } from 'node:child_process'
import { createServer } from 'node:net'
import { join } from 'node:path'

import { expect, test } from 'vitest'

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as { port: number }
      server.close((err) => (err ? reject(err) : resolve(port)))
    })
  })
}

test('cli.js surfaces the shared parser errors for unknown flags', async () => {
  const child = spawn(process.execPath, [join(process.cwd(), 'cli.js'), '--bogus'], {
    stdio: ['ignore', 'ignore', 'pipe'],
  })
  let stderr = ''
  child.stderr!.on('data', (chunk) => (stderr += chunk))
  const code = await new Promise<number | null>((resolve) => child.on('close', resolve))
  expect(code).toBe(1)
  expect(stderr).toContain('unexpected argument')
  expect(stderr).toContain('--bogus')
})

test('cli.js boots the gateway and answers /health', async () => {
  const port = await freePort()
  const child = spawn(
    process.execPath,
    [join(process.cwd(), 'cli.js'), '--host', '127.0.0.1', '--port', String(port)],
    { stdio: 'ignore' },
  )
  try {
    let health: Response | undefined
    for (let i = 0; i < 50 && !health; i++) {
      health = await fetch(`http://127.0.0.1:${port}/health`).catch(() => undefined)
      if (!health) await new Promise((r) => setTimeout(r, 100))
    }
    expect(health, 'gateway came up').toBeDefined()
    expect(health!.status).toBe(200)
    const body = await health!.json()
    expect(body.status).toBe('ok')
  } finally {
    child.kill()
  }
}, 15_000)
