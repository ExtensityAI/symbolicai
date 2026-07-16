from enum import StrEnum

from symai.runtime.errors import UnsupportedFeatureError
from symai.runtime.models import (
    ImageContent,
    LanguageModelRequest,
    LanguageModelSpec,
    ReasoningField,
    SamplingField,
)


def validate_language_model_capabilities(
    request: LanguageModelRequest,
    *,
    spec: LanguageModelSpec,
    provider: str,
    model: str,
) -> None:
    if not spec.vision and any(
        isinstance(content, ImageContent)
        for message in request.messages
        for content in message.content
    ):
        msg = f"{provider} model {model} does not support image input"
        raise UnsupportedFeatureError(msg)

    reasoning = request.reasoning
    if reasoning is not None:
        reasoning_fields = (
            (ReasoningField.ENABLED, reasoning.enabled),
            (ReasoningField.EFFORT, reasoning.effort),
            (ReasoningField.SUMMARY, reasoning.summary),
            (ReasoningField.FORMAT, reasoning.format),
            (ReasoningField.CLEAR, reasoning.clear),
        )
        for field, value in reasoning_fields:
            if value is not None and field not in spec.reasoning_fields:
                msg = f"{provider} model {model} does not support reasoning field {field.value}"
                raise UnsupportedFeatureError(msg)

        _validate_reasoning_value(
            reasoning.effort,
            supported=spec.reasoning_efforts,
            field=ReasoningField.EFFORT,
            provider=provider,
            model=model,
        )
        _validate_reasoning_value(
            reasoning.summary,
            supported=spec.reasoning_summaries,
            field=ReasoningField.SUMMARY,
            provider=provider,
            model=model,
        )
        _validate_reasoning_value(
            reasoning.format,
            supported=spec.reasoning_formats,
            field=ReasoningField.FORMAT,
            provider=provider,
            model=model,
        )

    sampling = request.sampling
    sampling_fields = (
        (SamplingField.MAX_TOKENS, sampling.max_tokens),
        (SamplingField.TEMPERATURE, sampling.temperature),
        (SamplingField.TOP_P, sampling.top_p),
        (SamplingField.STOP, sampling.stop or None),
        (SamplingField.SEED, sampling.seed),
        (SamplingField.FREQUENCY_PENALTY, sampling.frequency_penalty),
        (SamplingField.PRESENCE_PENALTY, sampling.presence_penalty),
    )
    for field, value in sampling_fields:
        if value is not None and field not in spec.sampling_fields:
            msg = f"{provider} model {model} does not support sampling field {field.value}"
            raise UnsupportedFeatureError(msg)

    if sampling.max_tokens is not None and sampling.max_tokens > spec.response_tokens:
        msg = f"{provider} model {model} supports at most {spec.response_tokens} output tokens"
        raise UnsupportedFeatureError(msg)


def _validate_reasoning_value[ValueT: StrEnum](
    value: ValueT | None,
    *,
    supported: tuple[ValueT, ...],
    field: ReasoningField,
    provider: str,
    model: str,
) -> None:
    if value is None or value in supported:
        return

    msg = f"{provider} model {model} does not support reasoning {field.value} {value.value}"
    raise UnsupportedFeatureError(msg)
