import pytest
from pydantic import ValidationError

from symai.backend.integrations.cerebras.chat import (
    ChatRequest,
    ChatResponse,
    JsonSchemaSpec,
    Message,
    ReasoningEffort,
    ResponseFormat,
    Role,
    Usage,
)


def _message() -> Message:
    return Message(role=Role.USER, content="hello")


# --- Bounds -----------------------------------------------------------------


def test_temperature_above_upper_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", temperature=2.1)


def test_temperature_below_lower_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", temperature=-0.1)


def test_temperature_within_bounds_succeeds():
    request = ChatRequest(messages=(_message(),), model="gpt-oss-120b", temperature=1.5)

    assert request.temperature == 1.5


def test_top_p_above_upper_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", top_p=1.1)


def test_top_p_below_lower_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", top_p=-0.1)


def test_max_completion_tokens_zero_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", max_completion_tokens=0)


def test_max_completion_tokens_negative_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", max_completion_tokens=-5)


def test_max_completion_tokens_none_succeeds():
    request = ChatRequest(
        messages=(_message(),), model="gpt-oss-120b", max_completion_tokens=None
    )

    assert request.max_completion_tokens is None


# --- Non-empty messages -------------------------------------------------------


def test_empty_messages_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(), model="gpt-oss-120b")


# --- Strict + forbid ----------------------------------------------------------


def test_temperature_string_raises_without_coercion():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", temperature="hot")


def test_unknown_field_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model="gpt-oss-120b", foo=1)


# --- Defaults -------------------------------------------------------------------


def test_minimal_chat_request_defaults():
    request = ChatRequest(messages=(_message(),), model="gpt-oss-120b")

    assert request.temperature == 1
    assert request.top_p == 1
    assert request.max_completion_tokens is None
    assert request.seed is None
    assert request.stop is None
    assert request.reasoning_effort is None
    assert request.response_format is None


# --- Open model IDs and server-owned reasoning capabilities -----------------------


def test_arbitrary_model_id_serializes_unchanged():
    model = "dedicated/acme-deployment-2026-07-10"
    request = ChatRequest(messages=(_message(),), model=model)

    assert request.model_dump()["model"] == model


def test_unsupported_model_reasoning_combination_is_not_rejected_locally():
    request = ChatRequest(
        messages=(_message(),),
        model="gpt-oss-120b",
        reasoning_effort=ReasoningEffort.NONE,
    )

    assert request.reasoning_effort == ReasoningEffort.NONE


# --- Stop ------------------------------------------------------------------------


def test_scalar_stop_serializes_as_string():
    request = ChatRequest(messages=(_message(),), model="custom-model", stop="STOP")

    assert request.model_dump()["stop"] == "STOP"


@pytest.mark.parametrize("stop", [(), ("one",), ("one", "two", "three", "four")])
def test_stop_sequence_with_at_most_four_items_serializes_as_tuple(
    stop: tuple[str, ...],
):
    request = ChatRequest(messages=(_message(),), model="custom-model", stop=stop)

    assert request.model_dump()["stop"] == stop


def test_stop_sequence_with_more_than_four_items_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=(_message(),),
            model="custom-model",
            stop=("one", "two", "three", "four", "five"),
        )


# --- JsonSchemaSpec alias --------------------------------------------------------


def test_json_schema_spec_populates_from_wire_alias():
    spec = JsonSchemaSpec.model_validate(
        {"name": "X", "schema": {"type": "object"}, "strict": True}
    )

    assert spec.json_schema_body == {"type": "object"}


def test_json_schema_spec_missing_name_raises():
    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"schema": {"type": "object"}})


def test_json_schema_spec_strict_defaults_to_false():
    spec = JsonSchemaSpec.model_validate({"name": "X", "schema": {"type": "object"}})

    assert spec.strict is False


def test_json_schema_spec_dump_by_alias_re_emits_schema_key():
    spec = JsonSchemaSpec(name="X", json_schema_body={"type": "object"})
    dumped = spec.model_dump(by_alias=True)

    assert "schema" in dumped
    assert "json_schema_body" not in dumped


def test_json_schema_spec_still_frozen_with_populate_by_name():
    spec = JsonSchemaSpec(name="X", json_schema_body={"type": "object"})

    with pytest.raises(ValidationError):
        spec.name = "Y"


def test_json_schema_spec_still_forbids_unknown_fields_with_populate_by_name():
    with pytest.raises(ValidationError):
        JsonSchemaSpec(name="X", json_schema_body={"type": "object"}, foo=1)


def test_json_schema_spec_still_strict_with_populate_by_name():
    with pytest.raises(ValidationError):
        JsonSchemaSpec(name="X", json_schema_body={"type": "object"}, strict="yes")


def test_json_schema_spec_rejects_non_json_values():
    with pytest.raises(ValidationError):
        JsonSchemaSpec(name="X", json_schema_body={"default": object()})


# --- ResponseFormat discriminant -----------------------------------------


def test_response_format_rejects_json_object_type():
    schema_spec = JsonSchemaSpec(name="X", json_schema_body={"type": "object"})

    with pytest.raises(ValidationError):
        ResponseFormat(type="json_object", json_schema=schema_spec)


# --- Full wire body ---------------------------------------------------------------


def test_full_chat_request_wire_body():
    schema_spec = JsonSchemaSpec(name="Answer", json_schema_body={"type": "object"}, strict=True)
    response_format = ResponseFormat(type="json_schema", json_schema=schema_spec)
    request = ChatRequest(
        messages=(Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="hi")),
        model="zai-glm-4.7",
        temperature=0.5,
        top_p=0.9,
        max_completion_tokens=100,
        seed=7,
        stop=("STOP",),
        reasoning_effort=ReasoningEffort.LOW,
        response_format=response_format,
    )

    dumped = request.model_dump(by_alias=True, exclude_none=True)

    # `messages`/`stop` are typed `tuple[X, ...]` on ChatRequest (repo convention for frozen
    # collection fields), so the python-mode dump preserves tuples; `model`/`reasoning_effort`
    # remain StrEnum members, which compare equal to their wire strings. httpx's JSON encoder
    # (Task 4) renders both tuples and StrEnum members identically to lists/strings on the wire.
    assert dumped == {
        "messages": (
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ),
        "model": "zai-glm-4.7",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_completion_tokens": 100,
        "seed": 7,
        "stop": ("STOP",),
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }
    assert isinstance(dumped["model"], str)
    assert isinstance(dumped["reasoning_effort"], str)


# --- Optional omission (exclude_none) ----------------------------------------------


def test_partial_chat_request_omits_unset_optionals():
    request = ChatRequest(
        messages=(_message(),),
        model="gpt-oss-120b",
        temperature=0.5,
    )

    dumped = request.model_dump(by_alias=True, exclude_none=True)

    for key in ("seed", "stop", "reasoning_effort", "response_format"):
        assert key not in dumped

    assert dumped["temperature"] == 0.5


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


# --- Tolerance (extra=allow) -----------------------------------------------------


def test_extra_top_level_and_nested_response_fields_are_preserved():
    response = ChatResponse.model_validate(
        {
            "id": "chatcmpl-123",
            "model": "gpt-oss-120b",
            "choices": [
                _choice_dict(
                    logprobs=None,
                    message={"role": "assistant", "content": "hello", "refusal": None},
                )
            ],
            "usage": _usage_dict(
                service_tier="default",
                completion_tokens_details={
                    "reasoning_tokens": 7,
                    "provider_counter": 8,
                },
            ),
        }
    )

    assert response.model_extra == {"id": "chatcmpl-123", "model": "gpt-oss-120b"}
    assert response.choices[0].model_extra == {"logprobs": None}
    assert response.choices[0].message.model_extra == {"refusal": None}
    assert response.usage.model_extra == {"service_tier": "default"}
    assert response.usage.completion_tokens_details.model_extra == {"provider_counter": 8}

    dumped = response.model_dump()
    assert dumped["id"] == "chatcmpl-123"
    assert dumped["choices"][0]["logprobs"] is None
    assert dumped["choices"][0]["message"]["refusal"] is None
    assert dumped["usage"]["service_tier"] == "default"
    assert dumped["usage"]["completion_tokens_details"]["provider_counter"] == 8


# --- Required-field validation ---------------------------------------------------


def test_missing_usage_raises_validation_error():
    with pytest.raises(ValidationError):
        ChatResponse.model_validate({"choices": [_choice_dict()]})
