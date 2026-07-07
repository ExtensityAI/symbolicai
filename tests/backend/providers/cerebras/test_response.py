import pytest
from pydantic import ValidationError

from symai.backend.providers.cerebras.response import ChatResponse, Usage


def _usage_dict(**overrides) -> dict:
    payload = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    payload.update(overrides)
    return payload


def _choice_dict(**overrides) -> dict:
    payload = {
        "index": 0,
        "message": {"role": "assistant", "content": "hello"},
        "finish_reason": "stop",
    }
    payload.update(overrides)
    return payload


# --- Happy parse --------------------------------------------------------------


def test_full_completion_parses_expected_fields():
    response = ChatResponse.model_validate(
        {
            "choices": [_choice_dict()],
            "usage": _usage_dict(),
        }
    )

    assert isinstance(response.choices, tuple)
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.message.content == "hello"
    assert choice.finish_reason == "stop"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


# --- Reasoning field present/absent --------------------------------------------


def test_reasoning_field_present_populates():
    response = ChatResponse.model_validate(
        {
            "choices": [
                _choice_dict(message={"role": "assistant", "content": "hi", "reasoning": "because"})
            ],
            "usage": _usage_dict(),
        }
    )

    assert response.choices[0].message.reasoning == "because"


def test_reasoning_field_absent_is_none():
    response = ChatResponse.model_validate(
        {
            "choices": [_choice_dict()],
            "usage": _usage_dict(),
        }
    )

    assert response.choices[0].message.reasoning is None


# --- Null content ---------------------------------------------------------------


def test_null_content_parses_as_none():
    response = ChatResponse.model_validate(
        {
            "choices": [_choice_dict(message={"role": "assistant", "content": None})],
            "usage": _usage_dict(),
        }
    )

    assert response.choices[0].message.content is None


# --- Nested usage details --------------------------------------------------------


def test_completion_tokens_details_reasoning_tokens_present_populates():
    usage = Usage.model_validate(_usage_dict(completion_tokens_details={"reasoning_tokens": 42}))

    assert usage.completion_tokens_details.reasoning_tokens == 42


def test_completion_tokens_details_absent_is_none():
    usage = Usage.model_validate(_usage_dict())

    assert usage.completion_tokens_details is None


def test_completion_tokens_details_accepted_and_rejected_prediction_tokens_present_populates():
    usage = Usage.model_validate(
        _usage_dict(
            completion_tokens_details={
                "accepted_prediction_tokens": 3,
                "rejected_prediction_tokens": 1,
            }
        )
    )

    assert usage.completion_tokens_details.accepted_prediction_tokens == 3
    assert usage.completion_tokens_details.rejected_prediction_tokens == 1


def test_completion_tokens_details_accepted_and_rejected_prediction_tokens_absent_are_none():
    usage = Usage.model_validate(_usage_dict(completion_tokens_details={"reasoning_tokens": 42}))

    assert usage.completion_tokens_details.accepted_prediction_tokens is None
    assert usage.completion_tokens_details.rejected_prediction_tokens is None


def test_prompt_tokens_details_cached_tokens_present_populates():
    usage = Usage.model_validate(_usage_dict(prompt_tokens_details={"cached_tokens": 8}))

    assert usage.prompt_tokens_details.cached_tokens == 8


def test_prompt_tokens_details_absent_is_none():
    usage = Usage.model_validate(_usage_dict())

    assert usage.prompt_tokens_details is None


def test_image_tokens_present_populates():
    usage = Usage.model_validate(_usage_dict(image_tokens=4))

    assert usage.image_tokens == 4


def test_image_tokens_absent_is_none():
    usage = Usage.model_validate(_usage_dict())

    assert usage.image_tokens is None


def test_nested_usage_details_round_trip_via_chat_response_full_parse():
    response = ChatResponse.model_validate(
        {
            "choices": [_choice_dict()],
            "usage": _usage_dict(
                completion_tokens_details={"reasoning_tokens": 7},
                prompt_tokens_details={"cached_tokens": 2},
                image_tokens=1,
            ),
        }
    )

    assert response.usage.completion_tokens_details.reasoning_tokens == 7
    assert response.usage.prompt_tokens_details.cached_tokens == 2
    assert response.usage.image_tokens == 1


# --- Tolerance (extra=ignore) ----------------------------------------------------


def test_extra_top_level_and_choice_keys_are_ignored():
    response = ChatResponse.model_validate(
        {
            "id": "chatcmpl-123",
            "model": "gpt-oss-120b",
            "created": 1700000000,
            "object": "chat.completion",
            "choices": [_choice_dict(logprobs=None)],
            "usage": _usage_dict(),
        }
    )

    assert response.model_dump().keys() == {"choices", "usage"}
    assert "logprobs" not in response.choices[0].model_dump()
    assert "id" not in response.model_dump()
    assert "model" not in response.model_dump()


# --- Required-field validation ---------------------------------------------------


def test_missing_usage_raises_validation_error():
    with pytest.raises(ValidationError):
        ChatResponse.model_validate({"choices": [_choice_dict()]})
