"""The capability gate: the single place a request is checked against a model spec."""

import pytest

from symai.runtime.capability import validate_language_model_capabilities
from symai.runtime.errors import UnsupportedFeatureError
from symai.runtime.models import (
    ImageContent,
    LanguageModelRequest,
    LanguageModelSpec,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningField,
    SamplingConfig,
    SamplingField,
    UserMessage,
)


def test_capability_gate_rejects_image_without_vision() -> None:
    request = LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="https://example.com/image.png"),)),)
    )

    with pytest.raises(UnsupportedFeatureError, match="image input"):
        validate_language_model_capabilities(
            request,
            spec=LanguageModelSpec(response_tokens=100, vision=False),
            provider="TestProvider",
            model="test-model",
        )


def test_capability_gate_rejects_unsupported_reasoning_field() -> None:
    request = LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="image"),)),),
        reasoning=ReasoningConfig(enabled=True),
    )

    with pytest.raises(UnsupportedFeatureError, match="reasoning field enabled"):
        validate_language_model_capabilities(
            request,
            spec=LanguageModelSpec(
                response_tokens=100,
                reasoning_fields=(ReasoningField.EFFORT,),
                reasoning_efforts=(ReasoningEffort.LOW,),
                vision=True,
            ),
            provider="TestProvider",
            model="test-model",
        )


def test_capability_gate_rejects_unsupported_reasoning_value() -> None:
    request = LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="image"),)),),
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    )

    with pytest.raises(UnsupportedFeatureError, match="reasoning effort high"):
        validate_language_model_capabilities(
            request,
            spec=LanguageModelSpec(
                response_tokens=100,
                reasoning_fields=(ReasoningField.EFFORT,),
                reasoning_efforts=(ReasoningEffort.LOW,),
                vision=True,
            ),
            provider="TestProvider",
            model="test-model",
        )


def test_capability_gate_rejects_unsupported_sampling_field() -> None:
    request = LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="image"),)),),
        sampling=SamplingConfig(temperature=0.5),
    )

    with pytest.raises(UnsupportedFeatureError, match="sampling field temperature"):
        validate_language_model_capabilities(
            request,
            spec=LanguageModelSpec(
                response_tokens=100,
                sampling_fields=(SamplingField.MAX_TOKENS,),
                vision=True,
            ),
            provider="TestProvider",
            model="test-model",
        )


def test_capability_gate_rejects_response_token_ceiling() -> None:
    request = LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="image"),)),),
        sampling=SamplingConfig(max_tokens=101),
    )

    with pytest.raises(UnsupportedFeatureError, match="at most 100 output tokens"):
        validate_language_model_capabilities(
            request,
            spec=LanguageModelSpec(
                response_tokens=100,
                sampling_fields=(SamplingField.MAX_TOKENS,),
                vision=True,
            ),
            provider="TestProvider",
            model="test-model",
        )
