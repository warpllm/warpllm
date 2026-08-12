import { ChatStream as NativeChatStream, Client as NativeClient } from '../index.js'

import { throwFromWire } from './errors.js'
import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
} from './generated/types.js'

/** Constructor options. Mirrors Rust's `ClientConfig`. */
export interface WarpLLMOptions {
  baseUrl?: string
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
      this.native = new NativeClient(
        JSON.stringify({
          base_url: options.baseUrl,
          timeout_secs: options.timeout,
          stream_read_timeout_secs: options.streamReadTimeout,
        }),
      )
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
    // Runtime dispatch is on the VALUE, so behaviour is right even where the
    // overload above cannot resolve — a request built as a variable, or handed
    // through a helper that widened `stream` to `boolean`.
    if (request.stream === true) {
      try {
        return new ChatCompletionStream(
          await this.native.chatCompletionsStream(JSON.stringify(request)),
        )
      } catch (err) {
        throwFromWire(err)
      }
    }
    let raw: string
    try {
      raw = await this.native.chatCompletions(JSON.stringify(request))
    } catch (err) {
      throwFromWire(err)
    }
    return JSON.parse(raw) as CreateChatCompletionResponse
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
