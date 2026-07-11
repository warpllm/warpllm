import { Client as NativeClient } from '../index.js'

import { throwFromWire } from './errors.js'
import type { ChatCompletion, ChatCompletionCreateParams, WarpLLMOptions } from './types.js'

class Completions {
  constructor(private readonly native: NativeClient) {}

  async create(params: ChatCompletionCreateParams): Promise<ChatCompletion> {
    const request = {
      model: params.model,
      messages: params.messages,
      temperature: params.temperature,
      max_tokens: params.maxTokens,
      top_p: params.topP,
      stop: params.stop,
      stream: params.stream || undefined,
    }
    let raw: string
    try {
      raw = await this.native.chatCompletion(JSON.stringify(request))
    } catch (err) {
      throwFromWire(err)
    }
    return JSON.parse(raw) as ChatCompletion
  }
}

class Chat {
  readonly completions: Completions

  constructor(native: NativeClient) {
    this.completions = new Completions(native)
  }
}

/**
 * Model strings are `provider/model`, e.g. `"openai/gpt-4o"`. API keys fall
 * back to OPENAI_API_KEY; a provider's key is only required when a request
 * targets that provider.
 */
export class WarpLLM {
  readonly chat: Chat

  constructor(options: WarpLLMOptions = {}) {
    const config = {
      openai_api_key: options.openaiApiKey,
      base_url: options.baseUrl,
      timeout_secs: options.timeout,
    }
    let native: NativeClient
    try {
      native = new NativeClient(JSON.stringify(config))
    } catch (err) {
      throwFromWire(err)
    }
    this.chat = new Chat(native)
  }
}
