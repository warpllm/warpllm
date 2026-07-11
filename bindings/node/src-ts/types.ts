export interface WarpLLMOptions {
  /** Falls back to the OPENAI_API_KEY environment variable. */
  openaiApiKey?: string
  baseUrl?: string
  /** Request timeout in seconds (default 600, matching the OpenAI SDK). */
  timeout?: number
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatCompletionCreateParams {
  /** `provider/model`, e.g. `"openai/gpt-4o"`. */
  model: string
  messages: ChatMessage[]
  temperature?: number
  maxTokens?: number
  topP?: number
  stop?: string[]
  /** Not yet implemented; `true` rejects with code `not_implemented`. */
  stream?: boolean
}

// ---------------------------------------------------------------------------
// Response — field-for-field copy of the `chat.completion` object.
// Object names and field order mirror the upstream reference:
// https://developers.openai.com/api/reference/resources/chat
// ---------------------------------------------------------------------------

export interface ChatCompletion {
  id: string
  choices: Choice[]
  created: number
  /** Echoes the caller-supplied `provider/model` string. */
  model: string
  /** Always `"chat.completion"`. */
  object: string
  moderation?: Moderation
  service_tier?: 'auto' | 'default' | 'flex' | 'scale' | 'priority'
  /** Deprecated upstream but still returned; passed through as-is. */
  system_fingerprint?: string
  usage?: CompletionUsage
}

export interface Choice {
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call'
  index: number
  /**
   * Optional per the docs; some OpenAI-compatible backends emit an explicit
   * `"logprobs": null`, which the gateway normalizes to absent.
   */
  logprobs?: ChoiceLogprobs
  message: ChatCompletionMessage
}

/** Both arrays are required and non-nullable when `logprobs` is present. */
export interface ChoiceLogprobs {
  content: ChatCompletionTokenLogprob[]
  refusal: ChatCompletionTokenLogprob[]
}

export interface ChatCompletionTokenLogprob {
  token: string
  bytes: number[] | null
  logprob: number
  top_logprobs: TopLogprob[]
}

export interface TopLogprob {
  token: string
  bytes: number[] | null
  logprob: number
}

export interface ChatCompletionMessage {
  content: string | null
  refusal: string | null
  role: 'assistant'
  annotations?: Annotation[]
  audio?: ChatCompletionAudio
  /** Deprecated upstream in favor of `tool_calls`. */
  function_call?: FunctionCall
  tool_calls?: ChatCompletionMessageToolCall[]
}

export interface Annotation {
  type: 'url_citation'
  url_citation: AnnotationURLCitation
}

export interface AnnotationURLCitation {
  end_index: number
  start_index: number
  title: string
  url: string
}

/** Deprecated upstream in favor of tool calls. */
export interface FunctionCall {
  /** JSON-encoded arguments; model-generated, so may be invalid JSON. */
  arguments: string
  name: string
}

/** Union discriminated by `type`. */
export type ChatCompletionMessageToolCall =
  | ChatCompletionMessageFunctionToolCall
  | ChatCompletionMessageCustomToolCall

export interface ChatCompletionMessageFunctionToolCall {
  id: string
  type: 'function'
  function: {
    /** JSON-encoded arguments; model-generated, so may be invalid JSON. */
    arguments: string
    name: string
  }
}

export interface ChatCompletionMessageCustomToolCall {
  id: string
  type: 'custom'
  custom: {
    input: string
    name: string
  }
}

export interface ChatCompletionAudio {
  id: string
  /** Base64-encoded audio bytes. */
  data: string
  expires_at: number
  transcript: string
}

export interface CompletionUsage {
  completion_tokens: number
  prompt_tokens: number
  total_tokens: number
  completion_tokens_details?: CompletionTokensDetails
  prompt_tokens_details?: PromptTokensDetails
}

export interface CompletionTokensDetails {
  accepted_prediction_tokens?: number
  audio_tokens?: number
  reasoning_tokens?: number
  rejected_prediction_tokens?: number
}

export interface PromptTokensDetails {
  audio_tokens?: number
  /** Unadjusted number of prompt tokens written to cache. */
  cache_write_tokens?: number
  cached_tokens?: number
}

/** Moderation results for the request input and the generated output. */
export interface Moderation {
  input: ModerationResults | Error
  output: ModerationResults | Error
}

// The docs define one shared ModerationResults/Error pair used by both
// `input` and `output`; both unions are discriminated by `type`.

export interface ModerationResults {
  type: 'moderation_results'
  model: string
  results: ModerationResult[]
}

/**
 * One verdict in `ModerationResults.results`. The docs leave this element
 * object unnamed; named here after its `type` string.
 */
export interface ModerationResult {
  categories: Record<string, boolean>
  category_applied_input_types: Record<string, string[]>
  category_scores: Record<string, number>
  flagged: boolean
  model: string
  type: 'moderation_result'
}

/**
 * Moderation error. Exact upstream name; it shadows the global `Error` type
 * in this module and wherever it is imported unaliased.
 */
export interface Error {
  type: 'error'
  code: string
  message: string
}
