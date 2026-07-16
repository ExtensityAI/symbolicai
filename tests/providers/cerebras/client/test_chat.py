import pytest
from pydantic import ValidationError

from symai.providers.cerebras.client.chat import (
    AssistantMessage,
    ChatCompletion,
    CreateChatCompletionRequest,
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
    UserMessage,
)


def _user_message() -> UserMessage:
    return UserMessage(role="user", content="hello")


def test_message_union_routes_raw_non_tool_roles_and_image_content():
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "dedicated/acme-model",
            "messages": (
                {"role": "system", "content": "system", "name": "policy"},
                {
                    "role": "developer",
                    "content": ({"type": "text", "text": "developer"},),
                },
                {
                    "role": "user",
                    "content": (
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ),
                    "name": "caller",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "prior reasoning",
                },
            ),
        }
    )

    assert isinstance(request.messages[0], SystemMessage)
    assert isinstance(request.messages[1], DeveloperMessage)
    assert isinstance(request.messages[1].content[0], TextContentPart)
    assert isinstance(request.messages[2], UserMessage)
    assert not isinstance(request.messages[2].content, str)
    assert isinstance(request.messages[2].content[1], ImageContentPart)
    assert isinstance(request.messages[2].content[1].image_url, ImageURL)
    assert isinstance(request.messages[3], AssistantMessage)
    assert request.messages[3].reasoning == "prior reasoning"


def test_message_union_rejects_tool_role():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": ({"role": "tool", "content": "result"},),
            }
        )


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_image_content_is_rejected_outside_user_messages(role: str):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (
                    {
                        "role": role,
                        "content": (
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ),
                    },
                ),
            }
        )


def test_complete_declared_request_serializes_with_aliases():
    request = CreateChatCompletionRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        clear_thinking=False,
        frequency_penalty=-0.5,
        max_completion_tokens=-1,
        prediction=Prediction(type="content", content="expected"),
        presence_penalty=0.5,
        prompt_cache_key="conversation-1",
        reasoning_effort=ReasoningEffort.HIGH,
        reasoning_format=ReasoningFormat.PARSED,
        response_format=JsonSchemaResponseFormat(
            type="json_schema",
            json_schema=JsonSchemaSpec(
                name="Answer",
                description="An answer",
                body={"type": "object"},
                strict=True,
            ),
        ),
        seed=0,
        service_tier=ServiceTier.DEFAULT,
        stop=(),
        temperature=0,
        top_p=1,
        user="user-1",
    )

    assert request.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-oss-120b",
        "clear_thinking": False,
        "frequency_penalty": -0.5,
        "max_completion_tokens": -1,
        "prediction": {"type": "content", "content": "expected"},
        "presence_penalty": 0.5,
        "prompt_cache_key": "conversation-1",
        "reasoning_effort": "high",
        "reasoning_format": "parsed",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "description": "An answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
        "seed": 0,
        "service_tier": "default",
        "stop": [],
        "temperature": 0,
        "top_p": 1,
        "user": "user-1",
    }


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "text"}, TextResponseFormat),
        ({"type": "json_object"}, JsonObjectResponseFormat),
        (
            {"type": "json_schema", "json_schema": {"name": "Answer"}},
            JsonSchemaResponseFormat,
        ),
    ],
)
def test_response_format_discriminator_routes_raw_payloads(
    payload: dict[str, object],
    expected_type: type[TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat],
):
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-oss-120b",
            "messages": ({"role": "user", "content": "hello"},),
            "response_format": payload,
        }
    )

    assert isinstance(request.response_format, expected_type)
    assert (
        request.model_dump(mode="json", by_alias=True)["response_format"]["type"] == payload["type"]
    )


def test_json_schema_defaults_and_aliases():
    from_wire = JsonSchemaSpec.model_validate({"name": "Answer", "schema": {"type": "object"}})
    from_python = JsonSchemaSpec(
        name="Answer",
        body={"type": "object"},
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
    field = "name"
    with pytest.raises(ValidationError):
        setattr(spec, field, "Other")


def test_unset_request_options_are_omitted():
    request = CreateChatCompletionRequest(model="gpt-oss-120b", messages=(_user_message(),))
    dumped = request.model_dump(mode="json", exclude_none=True)

    assert dumped == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-oss-120b",
    }


@pytest.mark.parametrize("value", [1, -1])
def test_max_completion_tokens_accepts_positive_and_documented_sentinel(
    value: int,
):
    request = CreateChatCompletionRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        max_completion_tokens=value,
    )
    assert request.max_completion_tokens == value


@pytest.mark.parametrize("value", [0, -2])
def test_max_completion_tokens_rejects_other_nonpositive_values(value: int):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(
            model="gpt-oss-120b",
            messages=(_user_message(),),
            max_completion_tokens=value,
        )


@pytest.mark.parametrize(
    "stop",
    ["END", (), ("A",), ("A", "B", "C", "D")],
)
def test_stop_forms_serialize(stop: str | tuple[str, ...]):
    request = CreateChatCompletionRequest(
        model="gpt-oss-120b",
        messages=(_user_message(),),
        stop=stop,
    )
    expected = list(stop) if isinstance(stop, tuple) else stop
    assert request.model_dump(mode="json")["stop"] == expected


def test_five_stop_sequences_are_rejected():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(
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
        ("prompt_cache_key", "x" * 1_025),
    ],
)
def test_request_field_bounds_are_enforced(field: str, value: object):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                field: value,
            }
        )


def test_messages_must_not_be_empty():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(messages=(), model="gpt-oss-120b")


def test_known_fields_remain_strict():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                "temperature": "1.0",
            }
        )


def test_arbitrary_model_id_and_reasoning_combination_are_accepted():
    request = CreateChatCompletionRequest(
        model="dedicated/acme-deployment",
        messages=(_user_message(),),
        reasoning_effort=ReasoningEffort.NONE,
        reasoning_format=ReasoningFormat.HIDDEN,
    )
    assert request.model == "dedicated/acme-deployment"
    assert request.reasoning_effort is ReasoningEffort.NONE


def test_json_compatible_unknown_request_extra_is_preserved():
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-oss-120b",
            "messages": (_user_message(),),
            "future_option": {"enabled": True},
        }
    )
    assert request.model_dump(mode="json")["future_option"] == {"enabled": True}


def test_non_json_unknown_request_extra_is_rejected():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                "future_option": object(),
            }
        )


def test_json_schema_value_rejects_non_json_objects():
    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"name": "Answer", "schema": {"value": object()}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logprobs", True),
        ("top_logprobs", 5),
        ("logit_bias", {"123": 1.0}),
    ],
)
def test_removed_logprob_request_fields_are_rejected(field: str, value: object):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "gpt-oss-120b",
                "messages": (_user_message(),),
                field: value,
            }
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


def test_null_content_parses_as_none():
    response = ChatCompletion.model_validate(
        {
            "choices": [_choice_dict(message={"role": "assistant", "content": None})],
            "usage": _usage_dict(),
        }
    )

    assert response.choices is not None
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is None


def test_complete_non_tool_response_fields_parse():
    payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {"content": [{"token": "A", "logprob": -0.1}]},
                "reasoning_logprobs": {"content": [{"token": "Think", "logprob": -0.2}]},
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "reasoning",
                },
            }
        ],
        "created": 1_700_000_000,
        "model": "gpt-oss-120b",
        "object": "chat.completion",
        "system_fingerprint": "fp-123",
        "service_tier": "auto",
        "service_tier_used": "priority",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "image_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {
                "accepted_prediction_tokens": 4,
                "rejected_prediction_tokens": 1,
                "reasoning_tokens": 2,
            },
        },
        "time_info": {
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
            "created": 1_700_000_000.5,
        },
    }

    response = ChatCompletion.model_validate(payload)

    assert response.model_dump(mode="json", exclude_none=True) == payload


def test_documented_response_fields_may_all_be_absent():
    response = ChatCompletion.model_validate({})

    assert response.id is None
    assert response.choices is None
    assert response.usage is None
    assert response.time_info is None


def test_partially_populated_nested_response_objects_parse():
    response = ChatCompletion.model_validate(
        {
            "choices": [{"message": {}}],
            "usage": {
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
            },
            "time_info": {},
        }
    )

    assert response.choices is not None
    assert response.choices[0].index is None
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is None
    assert response.usage is not None
    assert response.usage.total_tokens is None


def test_unknown_response_fields_survive_at_every_modeled_level():
    payload = {
        "future_top": 1,
        "choices": [
            {
                "future_choice": 2,
                "message": {"future_message": 3},
            }
        ],
        "usage": {
            "future_usage": 4,
            "prompt_tokens_details": {"future_prompt_detail": 5},
            "completion_tokens_details": {"future_completion_detail": 6},
        },
        "time_info": {"future_time": 7},
    }

    response = ChatCompletion.model_validate(payload)

    assert response.model_dump(mode="json", exclude_none=True) == payload


def test_non_object_chat_response_fails():
    with pytest.raises(ValidationError):
        ChatCompletion.model_validate([])
