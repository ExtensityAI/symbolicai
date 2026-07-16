from collections.abc import Sequence
from types import MappingProxyType
from typing import override

from pydantic import ValidationError

from symai.providers._client import errors as client_errors
from symai.providers._client.transport import APIResponse
from symai.providers._client.transport import ResponseMetadata as OpenAIResponseMetadata
from symai.providers._engine.base import ProviderEngine, retry_after_seconds
from symai.providers._engine.gate import validate_language_model_capabilities
from symai.providers._engine.mapping import ClientErrorMessages, raise_mapped_client_error
from symai.providers.openai.client import Client
from symai.providers.openai.client import responses as responses_api
from symai.runtime.errors import ErrorMetadata, InvalidResponseError
from symai.runtime.models import (
    AssistantMessage,
    AssistantOutputMessage,
    FinishReason,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelSpec,
    Message,
    ProviderId,
    ReasoningEffort,
    ReasoningField,
    ReasoningSummary,
    ResponseMetadata,
    SamplingField,
    TextContent,
    TokenUsage,
)

_HIGH_REASONING_EFFORT_MODELS: frozenset[responses_api.Model] = frozenset(
    {"gpt-5.5-pro", "gpt-5.4-pro", "o3-pro"}
)
_OPENAI_REASONING_FIELDS = (ReasoningField.EFFORT, ReasoningField.SUMMARY)
_OPENAI_REASONING_SUMMARIES = tuple(ReasoningSummary)
_OPENAI_BASE_SAMPLING_FIELDS = (SamplingField.MAX_TOKENS,)
_OPENAI_NONREASONING_SAMPLING_FIELDS = (
    *_OPENAI_BASE_SAMPLING_FIELDS,
    SamplingField.TEMPERATURE,
    SamplingField.TOP_P,
)


def _normalized_model_spec(spec: responses_api.ModelSpec) -> LanguageModelSpec:
    reasoning = spec.reasoning
    return LanguageModelSpec(
        response_tokens=spec.response_tokens,
        reasoning_fields=_OPENAI_REASONING_FIELDS if reasoning is not None else (),
        reasoning_efforts=tuple(ReasoningEffort(effort.value) for effort in reasoning.efforts)
        if reasoning is not None
        else (),
        reasoning_summaries=_OPENAI_REASONING_SUMMARIES if reasoning is not None else (),
        sampling_fields=_OPENAI_BASE_SAMPLING_FIELDS
        if reasoning is not None
        else _OPENAI_NONREASONING_SAMPLING_FIELDS,
        vision=spec.vision,
    )


MODEL_SPECS = MappingProxyType(
    {model: _normalized_model_spec(spec) for model, spec in responses_api.MODEL_SPECS.items()}
)

# Shared with the loader, which rejects an unsupported model before allocating transport.
UNSUPPORTED_MODEL_MESSAGE = "Unsupported OpenAI language model: {model}"

_ERROR_MESSAGES = ClientErrorMessages(
    authentication="OpenAI rejected authentication",
    rate_limit="OpenAI rate-limited the request",
    response="OpenAI returned an invalid response",
    transport="OpenAI transport failed",
    api="OpenAI API request failed with status {status_code}",
)


class ResponsesEngine(ProviderEngine[Client, responses_api.Model, LanguageModelSpec]):
    provider: ProviderId = "openai"

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
            response = self._client.create_response(provider_request)
        except client_errors.ClientError as error:
            raise_mapped_client_error(
                error,
                provider=self.provider,
                model=self.model,
                messages=_ERROR_MESSAGES,
            )

        return self._parse_response(response)

    def _build_request(self, request: LanguageModelRequest) -> responses_api.CreateResponseRequest:
        self._validate_request(request)
        return responses_api.CreateResponseRequest(
            input=tuple(self._input_message(message) for message in request.messages),
            model=self.model,
            max_output_tokens=request.sampling.max_tokens,
            metadata={label.key: label.value for label in request.metadata} or None,
            reasoning=self._reasoning_config(request),
            temperature=request.sampling.temperature,
            text=responses_api.TextConfig(format=self._response_format(request)),
            top_p=request.sampling.top_p,
            user=request.user,
        )

    def _validate_request(self, request: LanguageModelRequest) -> None:
        for message in request.messages:
            if isinstance(message, AssistantMessage) and message.reasoning is not None:
                self._unsupported("OpenAI does not accept normalized assistant reasoning input")

        if request.reasoning is not None and not self.model_spec.reasoning_fields:
            self._unsupported(f"OpenAI model {self.model} does not support reasoning")

        validate_language_model_capabilities(
            request,
            spec=self.model_spec,
            provider="OpenAI",
            model=self.model,
        )

    def _input_message(self, message: Message) -> responses_api.InputMessage:
        content: list[responses_api.InputContent] = []
        for part in message.content:
            if isinstance(part, TextContent):
                content.append(responses_api.InputText(type="input_text", text=part.text))
            else:
                content.append(
                    responses_api.InputImage(
                        type="input_image",
                        detail=part.detail.value if part.detail is not None else "auto",
                        image_url=part.url,
                    )
                )
        return responses_api.InputMessage(
            role=message.role,
            content=tuple(content),
        )

    def _response_format(self, request: LanguageModelRequest) -> responses_api.OutputFormat:
        response_format = request.response_format
        if isinstance(response_format, JsonSchemaResponseFormat):
            return responses_api.JsonSchemaFormat(
                type="json_schema",
                name=response_format.name,
                description=response_format.description,
                schema=response_format.json_schema,
                strict=response_format.strict,
            )
        if isinstance(response_format, JsonObjectResponseFormat):
            return responses_api.JsonObjectFormat(type="json_object")
        return responses_api.TextFormat(type="text")

    def _reasoning_config(
        self,
        request: LanguageModelRequest,
    ) -> responses_api.ReasoningConfig | None:
        if not self.model_spec.reasoning_fields:
            return None

        default_effort = (
            responses_api.ReasoningEffort.HIGH
            if self.model in _HIGH_REASONING_EFFORT_MODELS
            else responses_api.ReasoningEffort.MEDIUM
        )
        reasoning = request.reasoning
        effort = (
            responses_api.ReasoningEffort(reasoning.effort.value)
            if reasoning is not None and reasoning.effort is not None
            else default_effort
        )
        summary = (
            responses_api.ReasoningSummary(reasoning.summary.value)
            if reasoning is not None and reasoning.summary is not None
            else None
        )
        return responses_api.ReasoningConfig(effort=effort, summary=summary)

    def _parse_response(
        self, response: APIResponse[responses_api.Response, OpenAIResponseMetadata]
    ) -> LanguageModelResponse:
        raw = response.data
        error_metadata = self._execution_metadata(response)
        try:
            metadata = self._response_metadata(response)
        except ValidationError as error:
            msg = "OpenAI response metadata was invalid"
            raise InvalidResponseError(msg, metadata=error_metadata) from error
        finish_reason = self._finish_reason(raw, error_metadata)

        if any(isinstance(item, responses_api.CompactionOutput) for item in raw.output):
            msg = "OpenAI returned an unsupported output item"
            raise InvalidResponseError(msg, metadata=error_metadata)

        messages = [item for item in raw.output if isinstance(item, responses_api.OutputMessage)]
        for message in messages:
            self._validate_message_status(message, error_metadata, finish_reason)

        try:
            output = self._language_output(messages, raw, finish_reason)
            return LanguageModelResponse(outputs=(output,), metadata=metadata)
        except ValidationError as error:
            msg = "OpenAI response could not become a normalized language response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    def _language_output(
        self,
        messages: Sequence[responses_api.OutputMessage],
        raw: responses_api.Response,
        finish_reason: FinishReason,
    ) -> LanguageModelOutput:
        """Normalize one Responses result into a single output.

        The Responses API returns one logical answer as an ordered item list rather than
        N alternative completions, so every assistant message belongs to the same output.
        A response truncated while thinking carries reasoning and no message at all; its
        finish reason and usage still matter, so it must normalize rather than raise.
        Commentary-phase messages are the model thinking aloud, so they join the reasoning
        rather than contaminating the answer text.
        """
        answers = [message for message in messages if message.phase != "commentary"]
        commentary = [message for message in messages if message.phase == "commentary"]
        reasoning = self._reasoning_text(raw, commentary)

        text = tuple(
            TextContent(text=part.text)
            for message in answers
            for part in message.content
            if isinstance(part, responses_api.OutputText)
        )
        refusal = "".join(
            part.refusal
            for message in answers
            for part in message.content
            if isinstance(part, responses_api.Refusal)
        )
        return LanguageModelOutput(
            index=0,
            message=AssistantOutputMessage(content=text, reasoning=reasoning),
            refusal=refusal or None,
            finish_reason=finish_reason,
        )

    def _validate_message_status(
        self,
        message: responses_api.OutputMessage,
        error_metadata: ErrorMetadata,
        finish_reason: FinishReason,
    ) -> None:
        if message.status is responses_api.ItemStatus.COMPLETED:
            return
        if message.status is responses_api.ItemStatus.INCOMPLETE and finish_reason is not (
            FinishReason.STOP
        ):
            return

        msg = f"OpenAI assistant output status was {message.status.value!r}"
        raise InvalidResponseError(msg, metadata=error_metadata)

    def _finish_reason(
        self,
        response: responses_api.Response,
        error_metadata: ErrorMetadata,
    ) -> FinishReason:
        if response.status is responses_api.ResponseStatus.COMPLETED:
            return FinishReason.STOP
        if response.status is responses_api.ResponseStatus.FAILED:
            return FinishReason.ERROR
        if response.status is responses_api.ResponseStatus.INCOMPLETE:
            details = response.incomplete_details
            if details is not None and details.reason == "max_output_tokens":
                return FinishReason.LENGTH
            if details is not None and details.reason == "content_filter":
                return FinishReason.CONTENT_FILTER

        msg = f"OpenAI response status was {response.status.value!r}"
        raise InvalidResponseError(msg, metadata=error_metadata)

    def _reasoning_text(
        self,
        response: responses_api.Response,
        commentary: Sequence[responses_api.OutputMessage] = (),
    ) -> TextContent | None:
        parts: list[str] = []
        for output in response.output:
            if not isinstance(output, responses_api.ReasoningOutput):
                continue
            parts.extend(summary.text for summary in output.summary)
            parts.extend(content.text for content in output.content)
        parts.extend(
            part.text
            for message in commentary
            for part in message.content
            if isinstance(part, responses_api.OutputText)
        )
        return TextContent(text="\n".join(parts)) if parts else None

    def _response_metadata(
        self,
        response: APIResponse[responses_api.Response, OpenAIResponseMetadata],
    ) -> ResponseMetadata:
        raw = response.data
        normalized_usage = self._usage(response)
        return ResponseMetadata(
            provider=self.provider,
            requested_model=self.model,
            response_model=raw.model,
            status_code=response.metadata.status_code,
            request_id=response.metadata.request_id,
            retry_after=retry_after_seconds(response.metadata.retry_after),
            response_id=raw.id,
            created_at=raw.created_at,
            usage=normalized_usage,
        )

    def _usage(
        self,
        response: APIResponse[responses_api.Response, OpenAIResponseMetadata],
    ) -> TokenUsage | None:
        usage = response.data.usage
        if usage is None:
            return None
        if (
            usage.total_tokens != usage.input_tokens + usage.output_tokens
            or usage.input_tokens_details.cached_tokens > usage.input_tokens
            or usage.output_tokens_details.reasoning_tokens > usage.output_tokens
        ):
            return None

        try:
            return TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_prompt_tokens=usage.input_tokens_details.cached_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            )
        except ValidationError:
            return None

    def _execution_metadata(
        self,
        response: APIResponse[responses_api.Response, OpenAIResponseMetadata],
    ) -> ErrorMetadata:
        return self._error_metadata(response.metadata)

    def _error_metadata(self, metadata: OpenAIResponseMetadata) -> ErrorMetadata:
        request_id = metadata.request_id
        retry_after = retry_after_seconds(metadata.retry_after)
        return ErrorMetadata(
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            retry_after=retry_after,
        )
