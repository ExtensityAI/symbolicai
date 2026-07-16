from types import MappingProxyType
from typing import cast, override

from pydantic import ValidationError

from symai.providers._client import errors as client_errors
from symai.providers._engine.base import ProviderEngine, retry_after_seconds
from symai.providers._engine.gate import validate_language_model_capabilities
from symai.providers._engine.mapping import ClientErrorMessages, raise_mapped_client_error
from symai.providers.cerebras.client import Client
from symai.providers.cerebras.client import chat as chat_api
from symai.providers.cerebras.client.transport import APIResponse
from symai.providers.cerebras.client.transport import ResponseMetadata as CerebrasResponseMetadata
from symai.runtime.errors import ErrorMetadata, InvalidResponseError
from symai.runtime.models import (
    AssistantOutputMessage,
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
    ProviderId,
    RateLimitMetadata,
    ReasoningEffort,
    ReasoningField,
    ReasoningFormat,
    ResponseMetadata,
    SamplingField,
    SystemMessage,
    TextContent,
    TokenUsage,
    UserMessage,
)

_CEREBRAS_SAMPLING_FIELDS = tuple(SamplingField)


def _reasoning_fields(spec: chat_api.ReasoningSpec) -> tuple[ReasoningField, ...]:
    fields: list[ReasoningField] = []
    if spec.efforts:
        fields.append(ReasoningField.EFFORT)
    if spec.formats:
        fields.append(ReasoningField.FORMAT)
    if spec.clear_thinking:
        fields.append(ReasoningField.CLEAR)

    return tuple(fields)


def _normalized_model_spec(spec: chat_api.ModelSpec) -> LanguageModelSpec:
    reasoning = spec.reasoning
    if reasoning is None:
        return LanguageModelSpec(
            response_tokens=spec.response_tokens,
            sampling_fields=_CEREBRAS_SAMPLING_FIELDS,
            vision=spec.vision,
        )

    return LanguageModelSpec(
        response_tokens=spec.response_tokens,
        reasoning_fields=_reasoning_fields(reasoning),
        reasoning_efforts=tuple(ReasoningEffort(effort.value) for effort in reasoning.efforts),
        reasoning_formats=tuple(ReasoningFormat(value.value) for value in reasoning.formats),
        sampling_fields=_CEREBRAS_SAMPLING_FIELDS,
        vision=spec.vision,
    )


MODEL_SPECS = MappingProxyType(
    {model: _normalized_model_spec(spec) for model, spec in chat_api.MODEL_SPECS.items()}
)

# Shared with the loader, which rejects an unsupported model before allocating transport.
UNSUPPORTED_MODEL_MESSAGE = "Unsupported Cerebras language model: {model}"

_FINISH_REASONS = MappingProxyType(
    {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
)

_ERROR_MESSAGES = ClientErrorMessages(
    authentication="Cerebras rejected authentication",
    rate_limit="Cerebras rate-limited the request",
    response="Cerebras returned an invalid response",
    transport="Cerebras transport failed",
    api="Cerebras API request failed with status {status_code}",
)


class ChatCompletionsEngine(ProviderEngine[Client, chat_api.Model, LanguageModelSpec]):
    provider: ProviderId = "cerebras"

    @override
    def __init__(self, *, client: Client, model: str) -> None:
        super().__init__(
            client=client,
            model=model,
            model_specs=MODEL_SPECS,
            unsupported_model_message=UNSUPPORTED_MODEL_MESSAGE,
        )

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        provider_request = self._build_request(request)
        try:
            response = self._client.create_chat_completion(provider_request)
        except client_errors.ClientError as error:
            raise_mapped_client_error(
                error,
                provider=self.provider,
                model=self.model,
                messages=_ERROR_MESSAGES,
            )

        return self._parse_response(response)

    def _build_request(self, request: LanguageModelRequest) -> chat_api.CreateChatCompletionRequest:
        self._validate_request(request)
        reasoning = request.reasoning
        sampling = request.sampling
        return chat_api.CreateChatCompletionRequest(
            messages=tuple(self._message(message) for message in request.messages),
            model=self.model,
            clear_thinking=reasoning.clear if reasoning is not None else None,
            frequency_penalty=sampling.frequency_penalty,
            max_completion_tokens=sampling.max_tokens,
            presence_penalty=sampling.presence_penalty,
            reasoning_effort=(
                chat_api.ReasoningEffort(reasoning.effort.value)
                if reasoning is not None and reasoning.effort is not None
                else None
            ),
            reasoning_format=(
                chat_api.ReasoningFormat(reasoning.format.value)
                if reasoning is not None and reasoning.format is not None
                else None
            ),
            response_format=self._response_format(request),
            seed=sampling.seed,
            stop=sampling.stop or None,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            user=request.user,
        )

    def _validate_request(self, request: LanguageModelRequest) -> None:
        if request.metadata:
            self._unsupported("Cerebras does not support normalized request metadata")

        for message in request.messages:
            for content in message.content:
                if isinstance(content, ImageContent) and content.detail is not None:
                    self._unsupported("Cerebras does not support normalized image detail")

        validate_language_model_capabilities(
            request,
            spec=self.model_spec,
            provider="Cerebras",
            model=self.model,
        )

        sampling = request.sampling
        if len(sampling.stop) > 4:
            self._unsupported("Cerebras supports at most four stop sequences")

    def _message(self, message: Message) -> chat_api.Message:
        if isinstance(message, SystemMessage):
            content = tuple(
                chat_api.TextContentPart(type="text", text=part.text) for part in message.content
            )
            return chat_api.SystemMessage(role="system", content=content)
        if isinstance(message, DeveloperMessage):
            content = tuple(
                chat_api.TextContentPart(type="text", text=part.text) for part in message.content
            )
            return chat_api.DeveloperMessage(role="developer", content=content)
        if isinstance(message, UserMessage):
            content = tuple(self._content(part) for part in message.content)
            return chat_api.UserMessage(role="user", content=content)
        content = tuple(
            chat_api.TextContentPart(type="text", text=part.text) for part in message.content
        )
        return chat_api.AssistantMessage(
            role="assistant",
            content=content or None,
            reasoning=message.reasoning.text if message.reasoning is not None else None,
        )

    @staticmethod
    def _content(
        content: TextContent | ImageContent,
    ) -> chat_api.TextContentPart | chat_api.ImageContentPart:
        if isinstance(content, TextContent):
            return chat_api.TextContentPart(type="text", text=content.text)
        return chat_api.ImageContentPart(
            type="image_url",
            image_url=chat_api.ImageURL(url=content.url),
        )

    def _response_format(self, request: LanguageModelRequest) -> chat_api.ResponseFormat:
        response_format = request.response_format
        if isinstance(response_format, JsonSchemaResponseFormat):
            return chat_api.JsonSchemaResponseFormat(
                type="json_schema",
                json_schema=chat_api.JsonSchemaSpec(
                    name=response_format.name,
                    description=response_format.description,
                    body=response_format.json_schema,
                    strict=response_format.strict,
                ),
            )
        if isinstance(response_format, JsonObjectResponseFormat):
            return chat_api.JsonObjectResponseFormat(type="json_object")
        return chat_api.TextResponseFormat(type="text")

    def _parse_response(
        self,
        response: APIResponse[chat_api.ChatCompletion, CerebrasResponseMetadata],
    ) -> LanguageModelResponse:
        raw = response.data
        error_metadata = self._error_metadata(response.metadata)
        if not raw.choices:
            msg = "Cerebras response did not contain choices"
            raise InvalidResponseError(msg, metadata=error_metadata)

        try:
            metadata = self._response_metadata(response)
            seen_indices: set[int] = set()
            outputs: list[LanguageModelOutput] = []
            for choice in raw.choices:
                if choice.index is None:
                    msg = "Cerebras choice did not contain an index"
                    raise InvalidResponseError(msg, metadata=error_metadata)
                if choice.index in seen_indices:
                    msg = "Cerebras response contained duplicate choice indices"
                    raise InvalidResponseError(msg, metadata=error_metadata)
                seen_indices.add(choice.index)
                outputs.append(self._output(choice, error_metadata))
            outputs.sort(key=lambda output: output.index)
            return LanguageModelResponse(outputs=tuple(outputs), metadata=metadata)
        except (TypeError, ValidationError) as error:
            msg = "Cerebras response could not become a normalized language response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    def _output(
        self,
        choice: chat_api.Choice,
        error_metadata: ErrorMetadata,
    ) -> LanguageModelOutput:
        message = choice.message
        if message is None:
            msg = "Cerebras choice did not contain a message"
            raise InvalidResponseError(msg, metadata=error_metadata)
        if message.role is not None and message.role != "assistant":
            msg = "Cerebras choice message was not an assistant message"
            raise InvalidResponseError(msg, metadata=error_metadata)
        finish_reason_value = choice.finish_reason
        if finish_reason_value is None or finish_reason_value not in _FINISH_REASONS:
            msg = "Cerebras response contained an unsupported finish reason"
            raise InvalidResponseError(msg, metadata=error_metadata)
        finish_reason = _FINISH_REASONS[finish_reason_value]

        return LanguageModelOutput(
            index=cast("int", choice.index),
            message=AssistantOutputMessage(
                content=(TextContent(text=message.content),) if message.content is not None else (),
                reasoning=TextContent(text=message.reasoning) if message.reasoning else None,
            ),
            finish_reason=finish_reason,
        )

    def _response_metadata(
        self,
        response: APIResponse[chat_api.ChatCompletion, CerebrasResponseMetadata],
    ) -> ResponseMetadata:
        raw = response.data
        usage = self._usage(raw.usage)
        rate_limit = self._rate_limit(response.metadata)
        return ResponseMetadata(
            provider=self.provider,
            requested_model=self.model,
            response_model=raw.model,
            status_code=response.metadata.status_code,
            request_id=response.metadata.request_id,
            retry_after=retry_after_seconds(response.metadata.retry_after),
            response_id=raw.id,
            created_at=raw.created,
            system_fingerprint=raw.system_fingerprint,
            usage=usage,
            rate_limit=rate_limit,
        )

    @staticmethod
    def _rate_limit(
        metadata: CerebrasResponseMetadata,
    ) -> RateLimitMetadata | None:
        rate_limit = metadata.rate_limit
        values = (
            rate_limit.limit_requests_day,
            rate_limit.limit_tokens_minute,
            rate_limit.remaining_requests_day,
            rate_limit.remaining_tokens_minute,
            rate_limit.reset_requests_day,
            rate_limit.reset_tokens_minute,
        )
        if all(value is None for value in values):
            return None
        return RateLimitMetadata(
            limit_requests_day=rate_limit.limit_requests_day,
            limit_tokens_minute=rate_limit.limit_tokens_minute,
            remaining_requests_day=rate_limit.remaining_requests_day,
            remaining_tokens_minute=rate_limit.remaining_tokens_minute,
            reset_requests_day=rate_limit.reset_requests_day,
            reset_tokens_minute=rate_limit.reset_tokens_minute,
        )

    @staticmethod
    def _usage(usage: chat_api.Usage | None) -> TokenUsage | None:
        if usage is None:
            return None
        if (
            usage.prompt_tokens is not None
            and usage.completion_tokens is not None
            and usage.total_tokens is not None
            and usage.total_tokens != usage.prompt_tokens + usage.completion_tokens
        ):
            return None

        prompt_details = usage.prompt_tokens_details
        completion_details = usage.completion_tokens_details
        cached = prompt_details.cached_tokens if prompt_details is not None else None
        reasoning = completion_details.reasoning_tokens if completion_details is not None else None
        accepted = (
            completion_details.accepted_prediction_tokens
            if completion_details is not None
            else None
        )
        rejected = (
            completion_details.rejected_prediction_tokens
            if completion_details is not None
            else None
        )
        if usage.prompt_tokens is not None and cached is not None and cached > usage.prompt_tokens:
            return None
        if usage.completion_tokens is not None:
            if reasoning is not None and reasoning > usage.completion_tokens:
                return None
            if (
                accepted is not None
                and rejected is not None
                and accepted + rejected > usage.completion_tokens
            ):
                return None

        try:
            return TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
                cached_prompt_tokens=cached or 0,
                reasoning_tokens=reasoning or 0,
                image_tokens=usage.image_tokens or 0,
                accepted_prediction_tokens=accepted or 0,
                rejected_prediction_tokens=rejected or 0,
            )
        except ValidationError:
            return None
