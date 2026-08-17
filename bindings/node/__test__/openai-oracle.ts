// Compile-only drift alarm: warpllm's wire shapes measured against OpenAI's
// own declarations — the request it accepts, the whole completion, and the
// streamed chunk.
//
// `openai` is a devDependency and an ORACLE, never a contract. Nothing here is
// re-exported, and the package is absent from the published dependency tree —
// warpllm's wire types are its own, and this file only asserts they still fit
// what the vendor says. The version is pinned exactly, so an upstream field
// arrives as a failure here rather than as a surprise in someone's stream.
//
// Two questions, asked separately:
//
//  1. Does everything OpenAI can emit FIT warpllm's shape? Assignability
//     answers that, and it is the one users feel: a field we typed too
//     narrowly makes their real traffic a type error. It is also what makes
//     "permissive superset" a checked claim rather than a description —
//     including the nulls, which is why every optional-and-nullable field is
//     `Option<Option<T>>` in Rust.
//  2. Does warpllm MODEL every field OpenAI models? Assignability cannot
//     answer that — an object with extra properties is still assignable — so
//     the key sets are compared directly. A field we never modelled still
//     reaches callers through `unknown_fields`, but it reaches them untyped.
//
// There are no exceptions to either. If one becomes necessary, it belongs
// here, spelled as a type: a deviation nobody can read is a deviation nobody
// weighed.
import type {
  ChatCompletion,
  ChatCompletionAllowedToolChoice,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ChatCompletionFunctionTool,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionNamedToolChoiceCustom,
  ChatCompletionToolChoiceOption as OpenAIToolChoiceOption,
} from 'openai/resources/chat/completions'
import type { FunctionDefinition } from 'openai/resources/shared'

import type {
  ChatCompletionMessageToolCallChunk,
  ChatCompletionRequestMessage,
  ChatCompletionResponseMessage,
  ChatCompletionStreamResponseDelta,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  Choice,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
  FunctionObject,
  StreamChoice,
} from '../src-ts/generated/types.js'

// Fails to compile unless `T` is `never` — an empty array would satisfy
// `Missing[]` no matter what `Missing` held, which is why this is a constraint
// and not a value.
type Nothing<T extends never> = T

type Missing<Upstream, Ours> = Exclude<keyof Upstream, keyof Ours>

// ---------------------------------------------------------------------------
// 1. Anything OpenAI emits, warpllm holds
// ---------------------------------------------------------------------------

export const acceptsAnOpenAICompletion: CreateChatCompletionResponse = {} as ChatCompletion
export const acceptsAnOpenAIChunk: CreateChatCompletionStreamResponse = {} as ChatCompletionChunk

// The request asks the question in the direction a caller feels it: a body
// built against the vendor's own SDK has to be one warpllm accepts. Every
// message role, every content part, every tool shape — including the ones
// warpllm does not model, which must reach the provider rather than fail to
// typecheck.
export const acceptsEveryMessageRole: ChatCompletionRequestMessage = {} as ChatCompletionMessageParam

// The exceptions below share one cause, and it is a limit of what a `.d.ts`
// can SAY rather than of what warpllm accepts. Every one of these shapes
// deserializes and re-emits verbatim in Rust, where the field is a
// `serde_json::Value`.
//
// `Value` generates as `JsonValue`, whose object case is the mapped type
// `{ [key in string]: JsonValue }`. TypeScript will not assign an `interface`
// to a mapped type (interfaces get no implicit index signature), and will not
// assign `Record<string, unknown>` to it either (`unknown` is not
// `JsonValue`). OpenAI's SDK declares both. So the assertions are narrowed to
// exactly the fields that land on a `Value`, and the rest stays checked —
// which is the point: this same check is what caught `stop` being modelled as
// a list when OpenAI's own examples pass a bare string.
type LandsOnArbitraryJson =
  // `parameters` on a function, and `schema` on a json_schema format.
  | 'tools'
  | 'response_format'
  // Plus the two tool-choice shapes warpllm holds only in its catch-all.
  | 'tool_choice'

export const acceptsAnOpenAIRequest: CreateChatCompletionRequest = {} as Omit<
  ChatCompletionCreateParams,
  LandsOnArbitraryJson
>

// ...and `tool_choice` is still checked over everything but those two shapes,
// rather than dropped whole.
export const acceptsEveryModelledToolChoice: ChatCompletionToolChoiceOption = {} as Exclude<
  OpenAIToolChoiceOption,
  ChatCompletionAllowedToolChoice | ChatCompletionNamedToolChoiceCustom
>

// ---------------------------------------------------------------------------
// 2. Anything OpenAI models, warpllm names
// ---------------------------------------------------------------------------

export type CompletionFieldsAreModelled = Nothing<
  Missing<ChatCompletion, CreateChatCompletionResponse>
>
export type CompletionChoiceFieldsAreModelled = Nothing<Missing<ChatCompletion.Choice, Choice>>
export type MessageFieldsAreModelled = Nothing<
  Missing<ChatCompletionMessage, ChatCompletionResponseMessage>
>

export type ChunkFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk, CreateChatCompletionStreamResponse>
>
export type ChunkChoiceFieldsAreModelled = Nothing<Missing<ChatCompletionChunk.Choice, StreamChoice>>
export type DeltaFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk.Choice.Delta, ChatCompletionStreamResponseDelta>
>
export type ToolCallFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk.Choice.Delta.ToolCall, ChatCompletionMessageToolCallChunk>
>

// The request is asked this question only where warpllm CHOSE to type a shape
// whole. `CreateChatCompletionRequest` itself is deliberately a partial typing
// — `n`, `seed`, `presence_penalty` and a dozen more reach the provider
// through the catch-all rather than through a field — so comparing its key set
// to the vendor's would assert something warpllm does not claim. A function
// tool is the opposite: it is modelled field for field, because translating it
// to another protocol means reading every part of it.
export type FunctionToolFieldsAreModelled = Nothing<
  Missing<ChatCompletionFunctionTool, ChatCompletionTool>
>
export type FunctionFieldsAreModelled = Nothing<Missing<FunctionDefinition, FunctionObject>>
