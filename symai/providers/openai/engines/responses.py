from math import isfinite
from types import MappingProxyType
from typing import cast

from pydantic import JsonValue, ValidationError

from symai.providers.openai.client import Client
from symai.providers.openai.client import errors as openai_errors
from symai.providers.openai.client import responses as responses_api
from symai.providers.openai.client.transport import APIResponse
from symai.providers.openai.client.transport import ResponseMetadata as OpenAIResponseMetadata
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
    AssistantMessage,
    AssistantOutputMessage,
    ContentType,
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
    ReasoningSummary,
    ResponseFormatType,
    ResponseMetadata,
    SamplingField,
    TextContent,
    TokenUsage,
    UserMessage,
)

_HIGH_REASONING_EFFORT_MODELS: frozenset[responses_api.Model] = frozenset(
    {"gpt-5.5-pro", "gpt-5.4-pro", "o3-pro"}
)
_ALL_MESSAGE_ROLES = tuple(MessageRole)
_ALL_RESPONSE_FORMATS = tuple(ResponseFormatType)
_OPENAI_REASONING_FIELDS = (ReasoningField.EFFORT, ReasoningField.SUMMARY)
_OPENAI_REASONING_SUMMARIES = tuple(ReasoningSummary)
_OPENAI_BASE_SAMPLING_FIELDS = (SamplingField.MAX_TOKENS, SamplingField.TOP_LOGPROBS)
_OPENAI_NONREASONING_SAMPLING_FIELDS = (
    *_OPENAI_BASE_SAMPLING_FIELDS,
    SamplingField.TEMPERATURE,
    SamplingField.TOP_P,
)


def _normalized_model_spec(spec: responses_api.ModelSpec) -> LanguageModelSpec:
    reasoning = spec.reasoning
    return LanguageModelSpec(
        context_tokens=spec.context_tokens,
        response_tokens=spec.response_tokens,
        message_roles=_ALL_MESSAGE_ROLES,
        content_types=(ContentType.TEXT, ContentType.IMAGE) if spec.vision else (ContentType.TEXT,),
        response_formats=_ALL_RESPONSE_FORMATS,
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


class ResponsesEngine:
    provider: ProviderId = "openai"

    def __init__(self, *, client: Client, model: str) -> None:
        try:
            try:
                model_spec = MODEL_SPECS[model]
            except KeyError as error:
                msg = f"Unsupported OpenAI language model: {model}"
                raise UnsupportedModelError(msg) from error

            self._client = client
            self._model: responses_api.Model = cast("responses_api.Model", model)
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
    def model(self) -> responses_api.Model:
        return self._model

    @property
    def model_spec(self) -> LanguageModelSpec:
        return self._model_spec

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        provider_request = self._build_request(request)
        try:
            response = self._client.create_response(provider_request)
        except openai_errors.AuthError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI rejected authentication"
            raise AuthenticationError(msg, metadata=metadata) from error
        except openai_errors.RateLimitError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI rate-limited the request"
            raise RateLimitError(msg, metadata=metadata) from error
        except openai_errors.ResponseError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI returned an invalid response"
            raise InvalidResponseError(msg, metadata=metadata) from error
        except openai_errors.TransportError as error:
            metadata = ErrorMetadata(provider=self.provider, model=self.model)
            msg = "OpenAI transport failed"
            raise TransportError(msg, metadata=metadata) from error
        except openai_errors.APIError as error:
            metadata = self._error_metadata(error.metadata)
            msg = f"OpenAI API request failed with status {error.metadata.status_code}"
            raise ExecutionError(msg, metadata=metadata) from error

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
            top_logprobs=request.sampling.top_logprobs,
            top_p=request.sampling.top_p,
            user=request.user,
        )

    def _validate_request(self, request: LanguageModelRequest) -> None:
        for message in request.messages:
            if isinstance(message, AssistantMessage) and message.reasoning is not None:
                self._unsupported("OpenAI does not accept normalized assistant reasoning input")
            if isinstance(message, UserMessage):
                has_image = any(isinstance(content, ImageContent) for content in message.content)
                if has_image and not self.model_spec.vision:
                    self._unsupported(f"OpenAI model {self.model} does not support image input")

        reasoning = request.reasoning
        if not self.model_spec.reasoning_fields:
            if reasoning is not None:
                self._unsupported(f"OpenAI model {self.model} does not support reasoning")
        elif reasoning is not None:
            if reasoning.enabled is not None:
                self._unsupported("OpenAI reasoning does not support the normalized enabled field")
            if reasoning.format is not None:
                self._unsupported("OpenAI reasoning does not support normalized reasoning format")
            if reasoning.clear is not None:
                self._unsupported("OpenAI reasoning does not support normalized reasoning clearing")
            if (
                reasoning.effort is not None
                and reasoning.effort not in self.model_spec.reasoning_efforts
            ):
                self._unsupported(
                    f"OpenAI model {self.model} does not support reasoning effort "
                    f"{reasoning.effort.value}"
                )
            if (
                reasoning.summary is not None
                and reasoning.summary not in self.model_spec.reasoning_summaries
            ):
                self._unsupported(
                    f"OpenAI model {self.model} does not support reasoning summary "
                    f"{reasoning.summary.value}"
                )

        sampling = request.sampling
        if (
            sampling.max_tokens is not None
            and sampling.max_tokens > self.model_spec.response_tokens
        ):
            self._unsupported(
                f"OpenAI model {self.model} supports at most "
                f"{self.model_spec.response_tokens} output tokens"
            )
        if (
            sampling.temperature is not None
            and SamplingField.TEMPERATURE not in self.model_spec.sampling_fields
        ):
            self._unsupported(f"OpenAI model {self.model} does not support temperature")
        if (
            sampling.top_p is not None
            and SamplingField.TOP_P not in self.model_spec.sampling_fields
        ):
            self._unsupported(f"OpenAI model {self.model} does not support top_p")
        if sampling.stop:
            self._unsupported("OpenAI Responses does not support normalized stop sequences")
        if sampling.seed is not None:
            self._unsupported("OpenAI Responses does not support normalized seed")
        if sampling.frequency_penalty is not None:
            self._unsupported("OpenAI Responses does not support normalized frequency penalty")
        if sampling.presence_penalty is not None:
            self._unsupported("OpenAI Responses does not support normalized presence penalty")
        if sampling.logprobs is not None:
            self._unsupported("OpenAI Responses does not support normalized logprobs toggle")
        if sampling.logit_bias:
            self._unsupported("OpenAI Responses does not support normalized logit bias")

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
                schema=cast("JsonValue", response_format.json_schema.to_builtin()),
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
        self, response: APIResponse[responses_api.Response]
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

        reasoning = self._reasoning_text(raw)
        messages = [item for item in raw.output if isinstance(item, responses_api.OutputMessage)]
        if reasoning is not None and len(messages) != 1:
            msg = "OpenAI reasoning output requires exactly one assistant message"
            raise InvalidResponseError(msg, metadata=error_metadata)

        try:
            outputs = tuple(
                self._language_output(
                    index,
                    message,
                    reasoning if index == 0 else None,
                    error_metadata,
                    finish_reason,
                )
                for index, message in enumerate(messages)
            )
            return LanguageModelResponse(outputs=outputs, metadata=metadata)
        except ValidationError as error:
            msg = "OpenAI response could not become a normalized language response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    def _language_output(
        self,
        index: int,
        message: responses_api.OutputMessage,
        reasoning: TextContent | None,
        error_metadata: ErrorMetadata,
        finish_reason: FinishReason,
    ) -> LanguageModelOutput:
        if message.status is not responses_api.ItemStatus.COMPLETED and not (
            message.status is responses_api.ItemStatus.INCOMPLETE
            and finish_reason is not FinishReason.STOP
        ):
            msg = f"OpenAI assistant output status was {message.status.value!r}"
            raise InvalidResponseError(msg, metadata=error_metadata)

        text = tuple(
            TextContent(text=part.text)
            for part in message.content
            if isinstance(part, responses_api.OutputText)
        )
        refusal = "".join(
            part.refusal for part in message.content if isinstance(part, responses_api.Refusal)
        )
        return LanguageModelOutput(
            index=index,
            message=AssistantOutputMessage(content=text, reasoning=reasoning),
            refusal=refusal or None,
            finish_reason=finish_reason,
        )

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

    def _reasoning_text(self, response: responses_api.Response) -> TextContent | None:
        parts: list[str] = []
        for output in response.output:
            if not isinstance(output, responses_api.ReasoningOutput):
                continue
            parts.extend(summary.text for summary in output.summary)
            parts.extend(content.text for content in output.content)
        return TextContent(text="\n".join(parts)) if parts else None

    def _response_metadata(
        self,
        response: APIResponse[responses_api.Response],
    ) -> ResponseMetadata:
        raw = response.data
        normalized_usage = self._usage(response)
        return ResponseMetadata(
            provider=self.provider,
            requested_model=self.model,
            response_model=raw.model,
            status_code=response.metadata.status_code,
            request_id=response.metadata.request_id,
            retry_after=self._retry_after(response.metadata.retry_after),
            response_id=raw.id,
            created_at=raw.created_at,
            usage=normalized_usage,
        )

    def _usage(
        self,
        response: APIResponse[responses_api.Response],
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
        response: APIResponse[responses_api.Response],
    ) -> ErrorMetadata:
        return self._error_metadata(response.metadata)

    def _error_metadata(self, metadata: OpenAIResponseMetadata) -> ErrorMetadata:
        request_id = metadata.request_id
        retry_after = self._retry_after(metadata.retry_after)
        return ErrorMetadata(
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            retry_after=retry_after,
        )

    @staticmethod
    def _retry_after(value: float | None) -> float | None:
        return value if value is not None and value >= 0 and isfinite(value) else None

    @staticmethod
    def _unsupported(message: str) -> None:
        raise UnsupportedFeatureError(message)
