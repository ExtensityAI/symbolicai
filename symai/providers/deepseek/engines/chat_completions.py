import re
from math import isfinite
from types import MappingProxyType
from typing import Never, cast

from pydantic import ValidationError

from symai.providers.deepseek.client import Client
from symai.providers.deepseek.client import chat as chat_api
from symai.providers.deepseek.client import errors as deepseek_errors
from symai.providers.deepseek.client.transport import APIResponse
from symai.providers.deepseek.client.transport import ResponseMetadata as DeepSeekResponseMetadata
from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import (
    AssistantOutputMessage,
    ContentType,
    DeveloperMessage,
    FinishReason,
    ImageContent,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelSpec,
    Message,
    MessageRole,
    ProviderId,
    ReasoningEffort,
    ReasoningField,
    ResponseFormatType,
    ResponseMetadata,
    SamplingField,
    SystemMessage,
    TextContent,
    TokenUsage,
    UserMessage,
)

_DEEPSEEK_MESSAGE_ROLES = (
    MessageRole.SYSTEM,
    MessageRole.USER,
    MessageRole.ASSISTANT,
)
_DEEPSEEK_RESPONSE_FORMATS = (
    ResponseFormatType.TEXT,
    ResponseFormatType.JSON_OBJECT,
)
_DEEPSEEK_REASONING_FIELDS = (
    ReasoningField.ENABLED,
    ReasoningField.EFFORT,
)
_DEEPSEEK_SAMPLING_FIELDS = (
    SamplingField.MAX_TOKENS,
    SamplingField.TEMPERATURE,
    SamplingField.TOP_P,
    SamplingField.STOP,
    SamplingField.LOGPROBS,
    SamplingField.TOP_LOGPROBS,
)
_USER_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def _normalized_model_spec(spec: chat_api.ModelSpec) -> LanguageModelSpec:
    reasoning = spec.reasoning
    reasoning_efforts = (
        tuple(ReasoningEffort(effort.value) for effort in reasoning.efforts)
        if reasoning is not None
        else ()
    )
    return LanguageModelSpec(
        context_tokens=spec.context_tokens,
        response_tokens=spec.response_tokens,
        message_roles=_DEEPSEEK_MESSAGE_ROLES,
        content_types=(ContentType.TEXT,),
        response_formats=_DEEPSEEK_RESPONSE_FORMATS,
        reasoning_fields=_DEEPSEEK_REASONING_FIELDS if reasoning is not None else (),
        reasoning_efforts=reasoning_efforts,
        sampling_fields=_DEEPSEEK_SAMPLING_FIELDS,
        vision=False,
    )


MODEL_SPECS = MappingProxyType(
    {model: _normalized_model_spec(spec) for model, spec in chat_api.MODEL_SPECS.items()}
)

_FINISH_REASONS = MappingProxyType(
    {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
        "insufficient_system_resource": FinishReason.ERROR,
    }
)


class ChatCompletionsEngine:
    provider: ProviderId = "deepseek"

    def __init__(self, *, client: Client, model: str) -> None:
        try:
            try:
                model_spec = MODEL_SPECS[model]
            except KeyError as error:
                msg = f"Unsupported DeepSeek language model: {model}"
                raise UnsupportedModelError(msg) from error

            self._client = client
            self._model: chat_api.Model = cast("chat_api.Model", model)
            self._model_spec = model_spec
            self._closed = False
        except BaseException as error:
            try:
                client.close()
            except BaseException as cleanup_error:
                error.add_note(f"Engine construction cleanup failed: {cleanup_error!r}")
            raise

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._client.close()

    @property
    def model(self) -> chat_api.Model:
        return self._model

    @property
    def model_spec(self) -> LanguageModelSpec:
        return self._model_spec

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        provider_request = self._build_request(request)
        try:
            response = self._client.create_chat_completion(provider_request)
        except deepseek_errors.AuthError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "DeepSeek rejected authentication"
            raise AuthenticationError(msg, metadata=metadata) from error
        except deepseek_errors.RateLimitError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "DeepSeek rate-limited the request"
            raise RateLimitError(msg, metadata=metadata) from error
        except deepseek_errors.ResponseError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "DeepSeek returned an invalid response"
            raise InvalidResponseError(msg, metadata=metadata) from error
        except deepseek_errors.TransportError as error:
            metadata = ErrorMetadata(provider=self.provider, model=self.model)
            msg = "DeepSeek transport failed"
            raise TransportError(msg, metadata=metadata) from error
        except deepseek_errors.APIError as error:
            metadata = self._error_metadata(error.metadata)
            msg = f"DeepSeek API request failed with status {error.metadata.status_code}"
            raise ExecutionError(msg, metadata=metadata) from error

        return self._parse_response(response)

    def _build_request(self, request: LanguageModelRequest) -> chat_api.CreateChatCompletionRequest:
        self._validate_request(request)
        reasoning = request.reasoning
        sampling = request.sampling
        return chat_api.CreateChatCompletionRequest(
            messages=tuple(self._message(message) for message in request.messages),
            model=self.model,
            thinking=(
                chat_api.Thinking(
                    type=(
                        chat_api.ThinkingType.ENABLED
                        if reasoning.enabled
                        else chat_api.ThinkingType.DISABLED
                    )
                )
                if reasoning is not None and reasoning.enabled is not None
                else None
            ),
            reasoning_effort=(
                chat_api.ReasoningEffort(reasoning.effort.value)
                if reasoning is not None and reasoning.effort is not None
                else None
            ),
            max_tokens=sampling.max_tokens,
            response_format=self._response_format(request),
            stop=sampling.stop or None,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            logprobs=sampling.logprobs,
            top_logprobs=sampling.top_logprobs,
            user_id=request.user,
        )

    def _validate_request(self, request: LanguageModelRequest) -> None:
        if request.metadata:
            self._unsupported("DeepSeek does not support normalized request metadata")
        if isinstance(request.response_format, JsonSchemaResponseFormat):
            self._unsupported("DeepSeek does not support JSON Schema response format")
        if request.user is not None and (
            len(request.user) > 512 or _USER_ID_PATTERN.fullmatch(request.user) is None
        ):
            self._unsupported(
                "DeepSeek user identifiers may contain only ASCII letters, digits, underscores, "
                "and hyphens and must not exceed 512 characters"
            )

        for message in request.messages:
            if isinstance(message, DeveloperMessage):
                self._unsupported("DeepSeek does not support developer messages")
            if any(isinstance(content, ImageContent) for content in message.content):
                self._unsupported("DeepSeek does not support image content")
            if (
                not isinstance(message, (SystemMessage, UserMessage))
                and message.reasoning is not None
            ):
                self._unsupported(
                    "DeepSeek does not support normalized assistant reasoning prefixes"
                )

        reasoning = request.reasoning
        if reasoning is not None:
            if reasoning.summary is not None:
                self._unsupported("DeepSeek reasoning does not support normalized summaries")
            if reasoning.format is not None:
                self._unsupported("DeepSeek reasoning does not support normalized formats")
            if reasoning.clear is not None:
                self._unsupported("DeepSeek reasoning does not support normalized clear behavior")
            if (
                reasoning.effort is not None
                and reasoning.effort not in self.model_spec.reasoning_efforts
            ):
                self._unsupported(
                    f"DeepSeek model {self.model} does not support reasoning effort "
                    f"{reasoning.effort.value}"
                )
            if reasoning.enabled is False and reasoning.effort is not None:
                self._unsupported(
                    "DeepSeek reasoning effort cannot be set when thinking is disabled"
                )

        sampling = request.sampling
        unsupported_sampling = (
            ("seed", sampling.seed),
            ("frequency_penalty", sampling.frequency_penalty),
            ("presence_penalty", sampling.presence_penalty),
        )
        for field, value in unsupported_sampling:
            if value is not None:
                self._unsupported(f"DeepSeek does not support normalized {field}")
        if sampling.logit_bias:
            self._unsupported("DeepSeek does not support normalized logit bias")
        if (request.reasoning is None or request.reasoning.enabled is not False) and (
            sampling.temperature is not None or sampling.top_p is not None
        ):
            self._unsupported(
                "DeepSeek ignores temperature and top_p unless thinking is explicitly disabled"
            )
        if (
            sampling.max_tokens is not None
            and sampling.max_tokens > self.model_spec.response_tokens
        ):
            self._unsupported(
                f"DeepSeek model {self.model} supports at most "
                f"{self.model_spec.response_tokens} output tokens"
            )
        if len(sampling.stop) > 16:
            self._unsupported("DeepSeek supports at most sixteen stop sequences")
        if sampling.top_logprobs is not None and sampling.logprobs is not True:
            self._unsupported("DeepSeek top_logprobs requires logprobs to be true")

    @staticmethod
    def _message(message: Message) -> chat_api.Message:
        parts: list[str] = []
        for part in message.content:
            if isinstance(part, ImageContent):
                ChatCompletionsEngine._unsupported("DeepSeek does not support image content")
            parts.append(part.text)
        content = "".join(parts)
        if isinstance(message, SystemMessage):
            return chat_api.SystemMessage(role="system", content=content)
        if isinstance(message, UserMessage):
            return chat_api.UserMessage(role="user", content=content)
        return chat_api.AssistantMessage(role="assistant", content=content)

    @staticmethod
    def _response_format(
        request: LanguageModelRequest,
    ) -> chat_api.ResponseFormat | None:
        if isinstance(request.response_format, JsonObjectResponseFormat):
            return chat_api.JsonObjectResponseFormat(type="json_object")
        return None

    def _parse_response(
        self,
        response: APIResponse[chat_api.ChatCompletion],
    ) -> LanguageModelResponse:
        raw = response.data
        error_metadata = self._error_metadata(response.metadata)
        if not raw.choices:
            msg = "DeepSeek response did not contain choices"
            raise InvalidResponseError(msg, metadata=error_metadata)

        try:
            metadata = self._response_metadata(response)
            seen_indices: set[int] = set()
            outputs: list[LanguageModelOutput] = []
            for choice in raw.choices:
                if choice.index < 0:
                    msg = "DeepSeek choice index was negative"
                    raise InvalidResponseError(msg, metadata=error_metadata)
                if choice.index in seen_indices:
                    msg = "DeepSeek response contained duplicate choice indices"
                    raise InvalidResponseError(msg, metadata=error_metadata)
                seen_indices.add(choice.index)
                outputs.append(self._output(choice, error_metadata))
            outputs.sort(key=lambda output: output.index)
            return LanguageModelResponse(outputs=tuple(outputs), metadata=metadata)
        except (TypeError, ValidationError) as error:
            msg = "DeepSeek response could not become a normalized language response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    @staticmethod
    def _output(
        choice: chat_api.Choice,
        error_metadata: ErrorMetadata,
    ) -> LanguageModelOutput:
        if choice.message.role != "assistant":
            msg = "DeepSeek choice message was not an assistant message"
            raise InvalidResponseError(msg, metadata=error_metadata)
        finish_reason_value = choice.finish_reason
        if finish_reason_value not in _FINISH_REASONS:
            msg = "DeepSeek response contained an unsupported finish reason"
            raise InvalidResponseError(msg, metadata=error_metadata)
        return LanguageModelOutput(
            index=choice.index,
            message=AssistantOutputMessage(
                content=(TextContent(text=choice.message.content),)
                if choice.message.content is not None
                else (),
                reasoning=(
                    TextContent(text=choice.message.reasoning_content)
                    if choice.message.reasoning_content
                    else None
                ),
            ),
            finish_reason=_FINISH_REASONS[finish_reason_value],
        )

    def _response_metadata(
        self,
        response: APIResponse[chat_api.ChatCompletion],
    ) -> ResponseMetadata:
        raw = response.data
        return ResponseMetadata(
            provider=self.provider,
            requested_model=self.model,
            response_model=raw.model,
            status_code=response.metadata.status_code,
            request_id=response.metadata.request_id,
            retry_after=self._retry_after(response.metadata.retry_after),
            response_id=raw.id,
            created_at=raw.created,
            system_fingerprint=raw.system_fingerprint,
            usage=self._usage(raw.usage),
        )

    @staticmethod
    def _usage(usage: chat_api.Usage) -> TokenUsage | None:
        if min(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) < 0:
            return None
        if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
            return None

        cache_hit = usage.prompt_cache_hit_tokens
        cache_miss = usage.prompt_cache_miss_tokens
        if cache_hit is not None and (cache_hit < 0 or cache_hit > usage.prompt_tokens):
            return None
        if cache_miss is not None and (cache_miss < 0 or cache_miss > usage.prompt_tokens):
            return None
        if (
            cache_hit is not None
            and cache_miss is not None
            and cache_hit + cache_miss != usage.prompt_tokens
        ):
            return None

        completion_details = usage.completion_tokens_details
        reasoning = completion_details.reasoning_tokens if completion_details is not None else None
        if reasoning is not None and (reasoning < 0 or reasoning > usage.completion_tokens):
            return None

        try:
            return TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cached_prompt_tokens=cache_hit or 0,
                cache_miss_prompt_tokens=cache_miss or 0,
                reasoning_tokens=reasoning or 0,
            )
        except ValidationError:
            return None

    def _error_metadata(self, metadata: DeepSeekResponseMetadata) -> ErrorMetadata:
        return ErrorMetadata(
            provider=self.provider,
            model=self.model,
            request_id=metadata.request_id,
            retry_after=self._retry_after(metadata.retry_after),
        )

    @staticmethod
    def _retry_after(value: float | None) -> float | None:
        return value if value is not None and value >= 0 and isfinite(value) else None

    @staticmethod
    def _unsupported(message: str) -> Never:
        raise UnsupportedFeatureError(message)
