// Compile-only compatibility check for the package's intentionally public
// type names. Rust may refactor internal DTO names without changing this API.
import { ChatCompletionStream } from '../src-ts/index.js'

import type {
  Annotation,
  AnnotationURLCitation,
  ChatCompletionAudio,
  ChatCompletionMessage,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageToolCall,
  ChatCompletionMessageToolCallChunk,
  ChatCompletionRequestMessage,
  ChatCompletionStreamResponseDelta,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
  DeltaFunctionCall,
  Error,
  ErrorBody,
  FunctionCall,
  Moderation,
  ModerationResult,
  ModerationResults,
  PromptTokensDetails,
  StreamChoice,
  ToolCallChunkFunction,
  TopLogprob,
} from '../src-ts/index.js'

export type PublishedTypeSurface = [
  Annotation,
  AnnotationURLCitation,
  ChatCompletionAudio,
  ChatCompletionMessage,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageToolCall,
  ChatCompletionMessageToolCallChunk,
  ChatCompletionRequestMessage,
  ChatCompletionStreamResponseDelta,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
  DeltaFunctionCall,
  Error,
  ErrorBody,
  FunctionCall,
  Moderation,
  ModerationResult,
  ModerationResults,
  PromptTokensDetails,
  StreamChoice,
  ToolCallChunkFunction,
  TopLogprob,
]

// This name was historically the union, not only the function variant.
export const customToolCall: ChatCompletionMessageToolCall = {
  id: 'call-1',
  type: 'provider_custom',
  custom: { name: 'shell', input: 'pwd' },
}

// Rust's request deserializer accepts explicit null for optional values.
export const nullableRequest: CreateChatCompletionRequest = {
  model: 'openai/gpt-5.6',
  messages: [{ role: 'critic', content: 'review this' }],
  temperature: null,
}

// The overload has to DISCRIMINATE, which no runtime test can prove: a
// `chatCompletions` that always returned the response type would still pass
// every assertion in chat.spec.ts, because the streaming object it actually
// returns is iterated dynamically. These are compile-time only.
declare const client: import('../src-ts/index.js').WarpLLM

// `stream: true` inline selects the streaming overload...
export async function streamingOverloadYieldsAStream(): Promise<string> {
  const stream = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
  })
  let text = ''
  for await (const chunk of stream) {
    text += chunk.choices[0]?.delta.content ?? ''
  }
  return text
}

// ...and its absence selects the whole reply, whose `choices[0].message` a
// chunk has no equivalent of. If the overloads ever collapse to one signature,
// exactly one of these two functions stops compiling.
export async function theDefaultOverloadYieldsAWholeReply(): Promise<string | null> {
  const completion = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: [{ role: 'user', content: 'hi' }],
  })
  return completion.choices[0]?.message.content ?? null
}

// An explicit `false` is still a whole reply, not a stream.
export async function streamFalseIsNotAStream(): Promise<string | null> {
  const completion = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: [{ role: 'user', content: 'hi' }],
    stream: false,
  })
  return completion.choices[0]?.message.content ?? null
}

// A `stream` the checker only knows as `boolean` is genuinely either, and the
// signature has to admit it. Promising a whole reply here would type-check
// `.choices[0].message` on an object that is a stream at runtime — a crash the
// compiler had every chance to catch. `@ts-expect-error` is the assertion: it
// FAILS TO COMPILE if the union ever collapses back to the response type.
export async function aNonLiteralStreamFlagIsEither(streaming: boolean): Promise<void> {
  const result = await client.chatCompletions({
    model: 'openai/gpt-5.6',
    messages: [{ role: 'user', content: 'hi' }],
    stream: streaming,
  })
  // @ts-expect-error the union must be narrowed before either half is reached
  void result.choices
  if (result instanceof ChatCompletionStream) {
    for await (const chunk of result) void chunk.choices[0]?.delta
  } else {
    void result.choices[0]?.message
  }
}

// The chunk type has to be nameable: a caller annotating what it collects from
// a stream needs the element type, not only the stream.
export function chunkTypesAreNameable(
  chunk: CreateChatCompletionStreamResponse,
): ChatCompletionStreamResponseDelta | undefined {
  const choice: StreamChoice | undefined = chunk.choices[0]
  return choice?.delta
}
