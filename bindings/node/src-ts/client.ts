import {
  ChatStream as NativeChatStream,
  Client as NativeClient,
  BalancedClient as NativeBalancedClient,
} from '../index.js'

import { throwFromWire } from './errors.js'
import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
} from './generated/types.js'

/** One declared provider. `{}` means: serve it, key from the environment. */
export interface ProviderOptions {
  /**
   * This provider's API key, in place of the environment variable its registry
   * entry names. For callers holding keys somewhere this process's environment
   * cannot reach; it wins over that variable when both have one.
   */
  apiKey?: string
}

/** Constructor options. Mirrors Rust's `ClientConfig`. */
export interface WarpLLMOptions {
  baseUrl?: string
  /**
   * A roster of your own, in the same schema as warpllm's built-in
   * `specs.yaml`, merged over it. How a self-hosted OpenAI-compatible server —
   * vLLM, TGI, Ollama, llama.cpp — becomes routable, with no key required:
   *
   * ```yaml
   * providers:
   *   local:
   *     base_url: "http://localhost:8000/v1"
   *     auth: none
   *     models:
   *       local/llama-3.3-70b:
   *         supported_apis:
   *           - {api: openai_compat_chat_completions}
   * ```
   *
   * The built-in providers survive the merge, so adding `local/` leaves
   * `openai/` exactly where it was. Reusing a built-in provider's name
   * replaces that provider whole, models included, and does so **silently**
   * here: warpllm warns over Rust's `tracing`, which this binding installs no
   * subscriber for. The file is read when the client is constructed, so a
   * roster that cannot be used throws there rather than failing a request
   * later. Unset falls back to the `WARPLLM_SPECS` environment variable, and
   * then to the built-in roster alone.
   */
  specsPath?: string
  /** Request timeout in seconds (default 600, matching the OpenAI SDK). */
  timeout?: number
  /**
   * Seconds a stream may go without a single byte before warpllm gives up on
   * it. Unset means never, which is the default.
   *
   * `timeout` is a total deadline and cannot tell a slow stream from a wedged
   * one; this bounds the gap between bytes and resets on every one. Set it
   * above the slowest time-to-first-token you expect, not merely above the
   * gap between chunks — the wait before the first chunk is a gap too.
   */
  streamReadTimeout?: number
  /**
   * Providers this client serves, keyed by registry name (`openai`,
   * `deepseek`, …). ABSENT — not empty — means every provider on warpllm's
   * roster, which is what every client did before this option existed.
   *
   * Declaring narrows what is read as well as what is routed: only the named
   * providers' environment variables are consulted, and a model under a
   * provider not listed here is refused before any request goes out. A name
   * the registry does not hold throws from the constructor.
   */
  providers?: Record<string, ProviderOptions>
}

interface NativeChatClient {
  chatCompletions(json: string): Promise<string>
  chatCompletionsStream(json: string): Promise<NativeChatStream>
}

function serializeConfig(options: WarpLLMOptions): string {
  return JSON.stringify({
    base_url: options.baseUrl,
    specs_path: options.specsPath,
    timeout_secs: options.timeout,
    stream_read_timeout_secs: options.streamReadTimeout,
    // Entries are rebuilt rather than passed through, because the key
    // inside one is renamed too. `undefined` when absent, so
    // `JSON.stringify` drops it and Rust sees "no declaration" — while
    // `{}` survives as `{}`, which is the different claim it is.
    providers:
      options.providers &&
      Object.fromEntries(
        Object.entries(options.providers).map(([name, entry]) => [
          name,
          { api_key: entry.apiKey },
        ]),
      ),
  })
}

// Runtime dispatch is on the VALUE, so behaviour is right even where the
// overload above cannot resolve — a request built as a variable, or handed
// through a helper that widened `stream` to `boolean`.
async function chatCompletionsImpl<T extends CreateChatCompletionRequest>(
  native: NativeChatClient,
  request: T,
): Promise<CreateChatCompletionResponse | ChatCompletionStream> {
  if (request.stream === true) {
    try {
      return new ChatCompletionStream(
        await native.chatCompletionsStream(JSON.stringify(request)),
      )
    } catch (err) {
      throw throwFromWire(err)
    }
  }
  let raw: string
  try {
    raw = await native.chatCompletions(JSON.stringify(request))
  } catch (err) {
    // `throw` here, not a bare call: `throwFromWire` is declared `never`, so
    // `raw` below is sound only as long as it never returns. `throw`ing its
    // result keeps that true even if the signature ever loosens to `void`,
    // at zero cost today.
    throw throwFromWire(err)
  }
  return JSON.parse(raw) as CreateChatCompletionResponse
}

/**
 * Model strings are `provider/model`, e.g. `"openai/gpt-5.6"`. API keys come
 * from the environment (`OPENAI_API_KEY`); a provider's key is only required
 * when a request targets that provider.
 */
export class WarpLLM {
  private readonly native: NativeClient

  constructor(options: WarpLLMOptions = {}) {
    try {
      this.native = new NativeClient(serializeConfig(options))
    } catch (err) {
      throwFromWire(err)
    }
  }

  /**
   * One method, mirroring Rust's `client.chat_completions(request)`.
   *
   * The request crosses verbatim — its fields are Rust's, so nothing here
   * renames them and nothing here has to learn a field warpllm gains.
   *
   * THREE signatures, because `stream` decides the return type and a caller
   * may not have told the checker what it is. A literal `true` streams; its
   * absence, `false`, or `null` is a whole reply; anything else — a `boolean`
   * variable, a value threaded through a helper — is genuinely either, and
   * says so. Narrowing a union is a nuisance; being handed a
   * `CreateChatCompletionResponse` that is a stream at runtime is a crash on
   * `.choices[0].message`, so the nuisance is the better half of the trade.
   *
   * Generic in the request so a parameter warpllm does not model still
   * type-checks: TypeScript skips excess-property checking when the target is
   * a type parameter, which is what lets `{ ...req, seed: 7 }` compile without
   * a cast and without an index signature on the generated declaration. An
   * index signature would make every `interface` in the official `openai`
   * package unassignable here, since TypeScript gives interfaces no implicit
   * one.
   */
  async chatCompletions<T extends CreateChatCompletionRequest & { stream: true }>(
    request: T,
  ): Promise<ChatCompletionStream>
  async chatCompletions<
    T extends CreateChatCompletionRequest & { stream?: false | null | undefined },
  >(request: T): Promise<CreateChatCompletionResponse>
  async chatCompletions<T extends CreateChatCompletionRequest>(
    request: T,
  ): Promise<CreateChatCompletionResponse | ChatCompletionStream>
  async chatCompletions<T extends CreateChatCompletionRequest>(
    request: T,
  ): Promise<CreateChatCompletionResponse | ChatCompletionStream> {
    return chatCompletionsImpl(this.native, request)
  }
}

/**
 * The chunks of one streamed reply.
 *
 * ```ts
 * const stream = await client.chatCompletions({ model, messages, stream: true })
 * for await (const chunk of stream) {
 *   process.stdout.write(chunk.choices[0]?.delta.content ?? '')
 * }
 * ```
 *
 * The iteration protocol lives here rather than in Rust: napi's `Generator` is
 * synchronous and cannot express `for await`, so the native side exposes a
 * handle and this supplies `Symbol.asyncIterator` over it.
 *
 * Errors raise the same classes a non-streamed call does. One that arrives
 * mid-iteration is terminal — whatever produced it also ended the stream.
 */
export class ChatCompletionStream implements AsyncIterable<CreateChatCompletionStreamResponse> {
  constructor(private readonly native: NativeChatStream) {}

  async *[Symbol.asyncIterator](): AsyncIterator<CreateChatCompletionStreamResponse> {
    for (;;) {
      let raw: string | null
      try {
        raw = await this.native.next()
      } catch (err) {
        throwFromWire(err)
      }
      // `null` is the stream ending. The `[DONE]` sentinel never reaches here:
      // it is framing, consumed by the transport in Rust.
      if (raw === null) return
      yield JSON.parse(raw) as CreateChatCompletionStreamResponse
    }
  }
}

/** One candidate in a weighted round-robin group. */
export interface BalancedCandidate {
  /** Model string, e.g. `"openai/gpt-5.6"`. */
  model: string
  /** Relative weight — higher means more requests routed here. */
  weight: number
}

/**
 * Load-balanced client. Distributes requests across candidates via weighted
 * round-robin. The request's `model` field is rewritten to the selected
 * candidate on each call.
 *
 * ```ts
 * const client = new WarpLLMBalanced(
 *   [
 *     { model: 'openai/gpt-5.6', weight: 3 },
 *     { model: 'deepseek/deepseek-v4-pro', weight: 1 },
 *   ],
 *   { providers: { openai: {}, deepseek: {} } },
 * )
 * ```
 */
export class WarpLLMBalanced {
  private readonly native: NativeBalancedClient

  constructor(candidates: BalancedCandidate[], options: WarpLLMOptions = {}) {
    // TypeScript's checker only helps a caller who is one; the object
    // crossing to Rust below is plain JS at runtime, so a caller assembling
    // it dynamically (or calling from plain JS at all) gets a message
    // naming what they actually did wrong, not a Rust-side JSON-offset error
    // about a document they never saw.
    candidates.forEach((candidate, i) => {
      if (candidate.model === undefined || candidate.weight === undefined) {
        throw new TypeError(`candidates[${i}] needs both 'model' and 'weight': ${JSON.stringify(candidate)}`)
      }
    })
    try {
      this.native = new NativeBalancedClient(
        serializeConfig(options),
        JSON.stringify(candidates),
      )
    } catch (err) {
      throwFromWire(err)
    }
  }

  async chatCompletions<T extends CreateChatCompletionRequest & { stream: true }>(
    request: T,
  ): Promise<ChatCompletionStream>
  async chatCompletions<
    T extends CreateChatCompletionRequest & { stream?: false | null | undefined },
  >(request: T): Promise<CreateChatCompletionResponse>
  async chatCompletions<T extends CreateChatCompletionRequest>(
    request: T,
  ): Promise<CreateChatCompletionResponse | ChatCompletionStream>
  async chatCompletions<T extends CreateChatCompletionRequest>(
    request: T,
  ): Promise<CreateChatCompletionResponse | ChatCompletionStream> {
    return chatCompletionsImpl(this.native, request)
  }
}
