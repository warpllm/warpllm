export class WarpLLMError extends Error {
  /** Stable machine-readable slug, e.g. `"provider_error"`. */
  code: string

  constructor(message: string, code = 'error') {
    super(message)
    this.name = new.target.name
    this.code = code
  }
}

export class InvalidRequestError extends WarpLLMError {
  constructor(message: string) {
    super(message, 'invalid_request')
  }
}

export class APIConnectionError extends WarpLLMError {
  provider?: string

  constructor(message: string, provider?: string) {
    super(message, 'connection_error')
    this.provider = provider
  }
}

export class APIStatusError extends WarpLLMError {
  status?: number
  provider?: string
  errorType?: string

  constructor(
    message: string,
    opts: { status?: number; provider?: string; errorType?: string; code?: string } = {},
  ) {
    super(message, opts.code ?? 'provider_error')
    this.status = opts.status
    this.provider = opts.provider
    this.errorType = opts.errorType
  }
}

export class AuthenticationError extends APIStatusError {}

export class RateLimitError extends APIStatusError {}

export class NotImplementedError extends WarpLLMError {
  constructor(message: string) {
    super(message, 'not_implemented')
  }
}

interface WireError {
  code?: string
  message?: string
  provider?: string
  status?: number
  error_type?: string | null
}

/** Translates the native layer's wire-format JSON message into typed errors. */
export function throwFromWire(err: unknown): never {
  const raw = err instanceof Error ? err.message : String(err)
  let wire: WireError
  try {
    wire = JSON.parse(raw) as WireError
  } catch {
    throw new WarpLLMError(raw)
  }

  const message = wire.message ?? raw
  switch (wire.code) {
    case 'not_implemented':
      throw new NotImplementedError(message)
    case 'invalid_request':
      throw new InvalidRequestError(message)
    case 'missing_api_key':
      throw new AuthenticationError(message, { provider: wire.provider, code: wire.code })
    case 'connection_error':
      throw new APIConnectionError(message, wire.provider)
    case 'provider_error': {
      const opts = {
        status: wire.status,
        provider: wire.provider,
        errorType: wire.error_type ?? undefined,
      }
      if (wire.status === 401) throw new AuthenticationError(message, opts)
      if (wire.status === 429) throw new RateLimitError(message, opts)
      throw new APIStatusError(message, opts)
    }
    default:
      throw new WarpLLMError(message, wire.code)
  }
}
