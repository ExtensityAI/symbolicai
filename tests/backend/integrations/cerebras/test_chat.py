import math

import pytest
from pydantic import ValidationError

from symai.backend.integrations.cerebras.chat import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    DeveloperMessage,
    ImageContentPart,
    ImageURL,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    JsonSchemaSpec,
    Prediction,
    ReasoningEffort,
    ReasoningFormat,
    ServiceTier,
    SystemMessage,
    TextContentPart,
    TextResponseFormat,
    Usage,
    UserMessage,
)


def _user_message() -> UserMessage:
    return UserMessage(role="user", content="hello")


def test_message_union_models_non_tool_roles_and_image_content():
    request = ChatRequest(
        model="dedicated/acme-model",
        messages=(
            SystemMessage(role="system", content="system", name="policy"),
            DeveloperMessage(
                role="developer",
                content=(TextContentPart(type="text", text="developer"),),
            ),
            UserMessage(
                role="user",
                content=(
                    TextContentPart(type="text", text="describe"),
                    ImageContentPart(
                        type="image_url",
                        image_url=ImageURL(url="data:image/png;base64,AAAA"),
                    ),
                ),
                name="caller",
            ),
            AssistantMessage(
                role="assistant",
                content=None,
                reasoning="prior reasoning",
            ),
        ),
    )

    dumped = request.model_dump(mode="json", exclude_none=True)
    assert [message["role"] for message in dumped["messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
    ]
    assert dumped["messages"][2]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    assert dumped["messages"][3]["reasoning"] == "prior reasoning"


def test_message_union_rejects_tool_role():
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": ({"role": "tool", "content": "result"},),
            }
        )


def test_complete_declared_request_serializes_with_aliases():
    response_format = JsonSchemaResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaSpec(
            name="Answer",
            description="An answer",
            json_schema_body={"type": "object"},
            strict=True,
        ),
    )
    request = ChatRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        clear_thinking=False,
        frequency_penalty=-0.5,
        logit_bias={"123": -100.0, "456": 100.0},
        logprobs=True,
        max_completion_tokens=-1,
        prediction=Prediction(type="content", content="expected"),
        presence_penalty=0.5,
        prompt_cache_key="conversation-1",
        reasoning_effort=ReasoningEffort.HIGH,
        reasoning_format=ReasoningFormat.PARSED,
        response_format=response_format,
        seed=0,
        service_tier=ServiceTier.DEFAULT,
        stop=(),
        temperature=0,
        top_logprobs=20,
        top_p=1,
        user="user-1",
    )

    dumped = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped["clear_thinking"] is False
    assert dumped["frequency_penalty"] == -0.5
    assert dumped["logit_bias"] == {"123": -100.0, "456": 100.0}
    assert dumped["logprobs"] is True
    assert dumped["max_completion_tokens"] == -1
    assert dumped["prediction"] == {"type": "content", "content": "expected"}
    assert dumped["presence_penalty"] == 0.5
    assert dumped["prompt_cache_key"] == "conversation-1"
    assert dumped["reasoning_effort"] == "high"
    assert dumped["reasoning_format"] == "parsed"
    assert dumped["seed"] == 0
    assert dumped["service_tier"] == "default"
    assert dumped["stop"] == []
    assert dumped["temperature"] == 0
    assert dumped["top_logprobs"] == 20
    assert dumped["top_p"] == 1
    assert dumped["user"] == "user-1"
    assert dumped["response_format"]["json_schema"]["schema"] == {
        "type": "object"
    }
    assert "json_schema_body" not in dumped["response_format"]["json_schema"]


@pytest.mark.parametrize(
    "response_format",
    [
        TextResponseFormat(type="text"),
        JsonObjectResponseFormat(type="json_object"),
        JsonSchemaResponseFormat(
            type="json_schema",
            json_schema=JsonSchemaSpec(name="Answer"),
        ),
    ],
)
def test_response_format_variants_round_trip(
    response_format: (
        TextResponseFormat
        | JsonObjectResponseFormat
        | JsonSchemaResponseFormat
    ),
):
    request = ChatRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        response_format=response_format,
    )
    assert request.model_dump(mode="json", by_alias=True)["response_format"][
        "type"
    ] == response_format.type


def test_json_schema_defaults_and_aliases():
    from_wire = JsonSchemaSpec.model_validate(
        {"name": "Answer", "schema": {"type": "object"}}
    )
    from_python = JsonSchemaSpec(
        name="Answer",
        json_schema_body={"type": "object"},
    )

    assert from_wire.strict is False
    assert from_python.model_dump(mode="json", by_alias=True) == {
        "name": "Answer",
        "description": None,
        "schema": {"type": "object"},
        "strict": False,
    }


def test_json_schema_name_is_required():
    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"schema": {"type": "object"}})


def test_json_schema_model_remains_strict_frozen_and_extra_forbidden():
    spec = JsonSchemaSpec(name="Answer")

    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"name": "Answer", "strict": "yes"})
    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"name": "Answer", "future": True})
    with pytest.raises(ValidationError):
        setattr(spec, "name", "Other")


def test_unset_request_options_are_omitted():
    request = ChatRequest(model="gpt-oss-120b", messages=(_user_message(),))
    dumped = request.model_dump(mode="json", exclude_none=True)

    assert dumped == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-oss-120b",
    }


@pytest.mark.parametrize("value", [1, -1])
def test_max_completion_tokens_accepts_positive_and_documented_sentinel(
    value: int,
):
    request = ChatRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        max_completion_tokens=value,
    )
    assert request.max_completion_tokens == value


@pytest.mark.parametrize("value", [0, -2])
def test_max_completion_tokens_rejects_other_nonpositive_values(value: int):
    with pytest.raises(ValidationError):
        ChatRequest(
            model="gpt-oss-120b",
            messages=(_user_message(),),
            max_completion_tokens=value,
        )


@pytest.mark.parametrize(
    "stop",
    ["END", (), ("A",), ("A", "B", "C", "D")],
)
def test_stop_forms_serialize(stop: str | tuple[str, ...]):
    request = ChatRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        stop=stop,
    )
    expected = list(stop) if isinstance(stop, tuple) else stop
    assert request.model_dump(mode="json")["stop"] == expected


def test_five_stop_sequences_are_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(
            model="gpt-oss-120b",
            messages=(_user_message(),),
            stop=("A", "B", "C", "D", "E"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("frequency_penalty", -2.1),
        ("frequency_penalty", 2.1),
        ("presence_penalty", -2.1),
        ("presence_penalty", 2.1),
        ("top_logprobs", -1),
        ("top_logprobs", 21),
        ("prompt_cache_key", "x" * 1_025),
    ],
)
def test_request_field_bounds_are_enforced(field: str, value: object):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                field: value,
            }
        )


def test_messages_must_not_be_empty():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(), model="gpt-oss-120b")


def test_known_fields_remain_strict():
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                "temperature": "1.0",
            }
        )


def test_arbitrary_model_id_and_reasoning_combination_are_accepted():
    request = ChatRequest(
        model="dedicated/acme-deployment",
        messages=(_user_message(),),
        reasoning_effort=ReasoningEffort.NONE,
        reasoning_format=ReasoningFormat.HIDDEN,
    )
    assert request.model == "dedicated/acme-deployment"
    assert request.reasoning_effort is ReasoningEffort.NONE


def test_json_compatible_unknown_request_extra_is_preserved():
    request = ChatRequest.model_validate(
        {
            "model": "gpt-oss-120b",
            "messages": (_user_message(),),
            "future_option": {"enabled": True},
        }
    )
    assert request.model_dump(mode="json")["future_option"] == {"enabled": True}


def test_non_json_unknown_request_extra_is_rejected():
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                "future_option": object(),
            }
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -100.1, 100.1])
def test_invalid_logit_bias_is_rejected(value: float):
    with pytest.raises(ValidationError):
        ChatRequest(
            model="gpt-oss-120b",
            messages=(_user_message(),),
            logit_bias={"123": value},
        )


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

    assert usage.completion_tokens_details is not None
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

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.accepted_prediction_tokens == 3
    assert usage.completion_tokens_details.rejected_prediction_tokens == 1


def test_completion_tokens_details_accepted_and_rejected_prediction_tokens_absent_are_none():
    usage = Usage.model_validate(_usage_dict(completion_tokens_details={"reasoning_tokens": 42}))

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.accepted_prediction_tokens is None
    assert usage.completion_tokens_details.rejected_prediction_tokens is None


def test_prompt_tokens_details_cached_tokens_present_populates():
    usage = Usage.model_validate(_usage_dict(prompt_tokens_details={"cached_tokens": 8}))

    assert usage.prompt_tokens_details is not None
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

    assert response.usage.completion_tokens_details is not None
    assert response.usage.prompt_tokens_details is not None
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
    assert response.usage.completion_tokens_details is not None
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
