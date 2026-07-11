/**
 * API-surface fidelity: our response types must carry the exact field names
 * of the official openai SDK types (a dev-only dependency, generated from
 * the same OpenAPI spec as the docs). Pure type-level checks — enforced by
 * `npm run typecheck`, no runtime. When an SDK bump adds or renames a field,
 * the compile error names it.
 *
 * openai-node namespaces nested types (ChatCompletion.Choice); field keys,
 * not type names, are the contract being checked here.
 */
import type {
  ChatCompletion as SDKChatCompletion,
  ChatCompletionAudio as SDKChatCompletionAudio,
  ChatCompletionMessage as SDKChatCompletionMessage,
  ChatCompletionMessageCustomToolCall as SDKCustomToolCall,
  ChatCompletionMessageFunctionToolCall as SDKFunctionToolCall,
  ChatCompletionTokenLogprob as SDKChatCompletionTokenLogprob,
} from 'openai/resources/chat/completions'
import type { CompletionUsage as SDKCompletionUsage } from 'openai/resources/completions'

import type {
  Annotation,
  AnnotationURLCitation,
  ChatCompletion,
  ChatCompletionAudio,
  ChatCompletionMessage,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  Error,
  FunctionCall,
  Moderation,
  ModerationResult,
  ModerationResults,
  PromptTokensDetails,
  TopLogprob,
} from '../src-ts/types.js'

/** Non-never results make the compile error name the offending keys. */
type MissingFrom<Ours, SDK> = Exclude<keyof SDK, keyof Ours>
type ExtraIn<Ours, SDK> = Exclude<keyof Ours, keyof SDK>
type Never<T extends never> = T

type _ChatCompletionMissing = Never<MissingFrom<ChatCompletion, SDKChatCompletion>>
type _ChatCompletionExtra = Never<ExtraIn<ChatCompletion, SDKChatCompletion>>

type _ChoiceMissing = Never<MissingFrom<Choice, SDKChatCompletion.Choice>>
type _ChoiceExtra = Never<ExtraIn<Choice, SDKChatCompletion.Choice>>

type _LogprobsMissing = Never<MissingFrom<ChoiceLogprobs, SDKChatCompletion.Choice.Logprobs>>
type _LogprobsExtra = Never<ExtraIn<ChoiceLogprobs, SDKChatCompletion.Choice.Logprobs>>

type _MessageMissing = Never<MissingFrom<ChatCompletionMessage, SDKChatCompletionMessage>>
type _MessageExtra = Never<ExtraIn<ChatCompletionMessage, SDKChatCompletionMessage>>

type _TokenLogprobMissing = Never<
  MissingFrom<ChatCompletionTokenLogprob, SDKChatCompletionTokenLogprob>
>
type _TokenLogprobExtra = Never<
  ExtraIn<ChatCompletionTokenLogprob, SDKChatCompletionTokenLogprob>
>

type _TopLogprobMissing = Never<
  MissingFrom<TopLogprob, SDKChatCompletionTokenLogprob.TopLogprob>
>
type _TopLogprobExtra = Never<ExtraIn<TopLogprob, SDKChatCompletionTokenLogprob.TopLogprob>>

type _AudioMissing = Never<MissingFrom<ChatCompletionAudio, SDKChatCompletionAudio>>
type _AudioExtra = Never<ExtraIn<ChatCompletionAudio, SDKChatCompletionAudio>>

type _AnnotationMissing = Never<MissingFrom<Annotation, SDKChatCompletionMessage.Annotation>>
type _AnnotationExtra = Never<ExtraIn<Annotation, SDKChatCompletionMessage.Annotation>>

type _URLCitationMissing = Never<
  MissingFrom<AnnotationURLCitation, SDKChatCompletionMessage.Annotation.URLCitation>
>
type _URLCitationExtra = Never<
  ExtraIn<AnnotationURLCitation, SDKChatCompletionMessage.Annotation.URLCitation>
>

type _FunctionCallMissing = Never<
  MissingFrom<FunctionCall, SDKChatCompletionMessage.FunctionCall>
>
type _FunctionCallExtra = Never<ExtraIn<FunctionCall, SDKChatCompletionMessage.FunctionCall>>

type _FunctionToolCallMissing = Never<
  MissingFrom<ChatCompletionMessageFunctionToolCall, SDKFunctionToolCall>
>
type _FunctionToolCallExtra = Never<
  ExtraIn<ChatCompletionMessageFunctionToolCall, SDKFunctionToolCall>
>
type _FunctionPayloadMissing = Never<
  MissingFrom<ChatCompletionMessageFunctionToolCall['function'], SDKFunctionToolCall.Function>
>
type _FunctionPayloadExtra = Never<
  ExtraIn<ChatCompletionMessageFunctionToolCall['function'], SDKFunctionToolCall.Function>
>

type _CustomToolCallMissing = Never<
  MissingFrom<ChatCompletionMessageCustomToolCall, SDKCustomToolCall>
>
type _CustomToolCallExtra = Never<ExtraIn<ChatCompletionMessageCustomToolCall, SDKCustomToolCall>>
type _CustomPayloadMissing = Never<
  MissingFrom<ChatCompletionMessageCustomToolCall['custom'], SDKCustomToolCall.Custom>
>
type _CustomPayloadExtra = Never<
  ExtraIn<ChatCompletionMessageCustomToolCall['custom'], SDKCustomToolCall.Custom>
>

type _UsageMissing = Never<MissingFrom<CompletionUsage, SDKCompletionUsage>>
type _UsageExtra = Never<ExtraIn<CompletionUsage, SDKCompletionUsage>>

type _PromptDetailsMissing = Never<
  MissingFrom<PromptTokensDetails, SDKCompletionUsage.PromptTokensDetails>
>
// Known skew: the docs and openai-python ship `cache_write_tokens`, but
// openai-node 6.45.0 does not yet. Drop the exemption once it appears.
type _PromptDetailsExtra = Never<
  Exclude<
    ExtraIn<PromptTokensDetails, SDKCompletionUsage.PromptTokensDetails>,
    'cache_write_tokens'
  >
>

type _CompletionDetailsMissing = Never<
  MissingFrom<CompletionTokensDetails, SDKCompletionUsage.CompletionTokensDetails>
>
type _CompletionDetailsExtra = Never<
  ExtraIn<CompletionTokensDetails, SDKCompletionUsage.CompletionTokensDetails>
>

type _ModerationMissing = Never<MissingFrom<Moderation, SDKChatCompletion.Moderation>>
type _ModerationExtra = Never<ExtraIn<Moderation, SDKChatCompletion.Moderation>>

// Like the docs (and unlike openai-python), openai-node uses one shared
// ModerationResults/Error set for moderation input and output.
type SDKModerationResults = SDKChatCompletion.Moderation.ModerationResults
type SDKModerationResult = SDKChatCompletion.Moderation.ModerationResults.Result
type SDKModerationError = SDKChatCompletion.Moderation.Error

type _ModResultsMissing = Never<MissingFrom<ModerationResults, SDKModerationResults>>
type _ModResultsExtra = Never<ExtraIn<ModerationResults, SDKModerationResults>>
type _ModResultMissing = Never<MissingFrom<ModerationResult, SDKModerationResult>>
type _ModResultExtra = Never<ExtraIn<ModerationResult, SDKModerationResult>>
type _ModErrorMissing = Never<MissingFrom<Error, SDKModerationError>>
type _ModErrorExtra = Never<ExtraIn<Error, SDKModerationError>>
