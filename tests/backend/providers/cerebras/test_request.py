import pytest
from pydantic import ValidationError

from symai.backend.providers.cerebras.request import (
    CerebrasResponseFormat,
    ChatRequest,
    JsonSchemaSpec,
    Message,
    Role,
)
from symai.backend.providers.cerebras.spec import CerebrasModel, ReasoningEffort


def _message() -> Message:
    return Message(role=Role.USER, content="hello")


# --- Bounds -----------------------------------------------------------------


def test_temperature_above_upper_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, temperature=2.1)


def test_temperature_below_lower_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, temperature=-0.1)


def test_temperature_within_bounds_succeeds():
    request = ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, temperature=1.5)

    assert request.temperature == 1.5


def test_top_p_above_upper_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, top_p=1.1)


def test_top_p_below_lower_bound_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, top_p=-0.1)


def test_max_completion_tokens_zero_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, max_completion_tokens=0
        )


def test_max_completion_tokens_negative_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, max_completion_tokens=-5
        )


def test_max_completion_tokens_none_succeeds():
    request = ChatRequest(
        messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, max_completion_tokens=None
    )

    assert request.max_completion_tokens is None


# --- Non-empty messages -------------------------------------------------------


def test_empty_messages_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(), model=CerebrasModel.GPT_OSS_120B)


# --- Strict + forbid ----------------------------------------------------------


def test_temperature_string_raises_without_coercion():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, temperature="hot")


def test_unknown_field_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B, foo=1)


# --- Defaults -------------------------------------------------------------------


def test_minimal_chat_request_defaults():
    request = ChatRequest(messages=(_message(),), model=CerebrasModel.GPT_OSS_120B)

    assert request.temperature == 1
    assert request.top_p == 1
    assert request.max_completion_tokens is None
    assert request.seed is None
    assert request.stop is None
    assert request.reasoning_effort is None
    assert request.response_format is None


# --- Reasoning-effort/model cross-check ------------------------------------------


def test_reasoning_effort_high_on_gpt_oss_succeeds():
    request = ChatRequest(
        messages=(_message(),),
        model=CerebrasModel.GPT_OSS_120B,
        reasoning_effort=ReasoningEffort.HIGH,
    )

    assert request.reasoning_effort == ReasoningEffort.HIGH


def test_reasoning_effort_none_on_gpt_oss_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=(_message(),),
            model=CerebrasModel.GPT_OSS_120B,
            reasoning_effort=ReasoningEffort.NONE,
        )


def test_reasoning_effort_none_on_zai_glm_succeeds():
    request = ChatRequest(
        messages=(_message(),),
        model=CerebrasModel.ZAI_GLM_4_7,
        reasoning_effort=ReasoningEffort.NONE,
    )

    assert request.reasoning_effort == ReasoningEffort.NONE


# --- JsonSchemaSpec alias --------------------------------------------------------


def test_json_schema_spec_populates_from_wire_alias():
    spec = JsonSchemaSpec.model_validate(
        {"name": "X", "schema": {"type": "object"}, "strict": True}
    )

    assert spec.json_schema_body == {"type": "object"}


def test_json_schema_spec_missing_name_raises():
    with pytest.raises(ValidationError):
        JsonSchemaSpec.model_validate({"schema": {"type": "object"}})


def test_json_schema_spec_strict_defaults_to_true():
    spec = JsonSchemaSpec.model_validate({"name": "X", "schema": {"type": "object"}})

    assert spec.strict is True


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


# --- CerebrasResponseFormat discriminant -----------------------------------------


def test_response_format_rejects_json_object_type():
    schema_spec = JsonSchemaSpec(name="X", json_schema_body={"type": "object"})

    with pytest.raises(ValidationError):
        CerebrasResponseFormat(type="json_object", json_schema=schema_spec)


# --- Full wire body ---------------------------------------------------------------


def test_full_chat_request_wire_body():
    schema_spec = JsonSchemaSpec(name="Answer", json_schema_body={"type": "object"}, strict=True)
    response_format = CerebrasResponseFormat(type="json_schema", json_schema=schema_spec)
    request = ChatRequest(
        messages=(Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="hi")),
        model=CerebrasModel.ZAI_GLM_4_7,
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
        model=CerebrasModel.GPT_OSS_120B,
        temperature=0.5,
    )

    dumped = request.model_dump(by_alias=True, exclude_none=True)

    for key in ("seed", "stop", "reasoning_effort", "response_format"):
        assert key not in dumped

    assert dumped["temperature"] == 0.5
