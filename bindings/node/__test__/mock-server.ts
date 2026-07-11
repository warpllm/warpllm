import { createServer, type Server } from 'node:http'

export interface RecordedRequest {
  method?: string
  url?: string
  headers: NodeJS.Dict<string | string[]>
  body: unknown
}

interface CannedResponse {
  status: number
  body: unknown
}

/** Minimal localhost HTTP mock: queue responses, record requests. */
export class MockServer {
  readonly requests: RecordedRequest[] = []
  private readonly queue: CannedResponse[] = []

  private constructor(
    private readonly server: Server,
    readonly url: string,
  ) {}

  static async start(): Promise<MockServer> {
    // Definite assignment: the handler only runs once requests arrive,
    // which is after `mock` is constructed below.
    let mock!: MockServer
    const server = createServer((req, res) => {
      let raw = ''
      req.on('data', (chunk: Buffer) => {
        raw += chunk.toString()
      })
      req.on('end', () => {
        mock.requests.push({
          method: req.method,
          url: req.url,
          headers: req.headers,
          body: raw ? JSON.parse(raw) : undefined,
        })
        const next = mock.queue.shift() ?? { status: 404, body: { error: 'no canned response' } }
        res.writeHead(next.status, { 'content-type': 'application/json' })
        res.end(JSON.stringify(next.body))
      })
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const address = server.address()
    if (address === null || typeof address === 'string') {
      throw new Error('expected a TCP address')
    }
    mock = new MockServer(server, `http://127.0.0.1:${address.port}`)
    return mock
  }

  respondWith(status: number, body: unknown): void {
    this.queue.push({ status, body })
  }

  async close(): Promise<void> {
    await new Promise<void>((resolve, reject) =>
      this.server.close((err) => (err ? reject(err) : resolve())),
    )
  }
}
