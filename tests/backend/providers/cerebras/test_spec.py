import dataclasses

import pytest

from symai.backend.providers.cerebras.spec import (
    CEREBRAS_MODEL_SPECS,
    CerebrasModel,
    ReasoningEffort,
)


@pytest.mark.parametrize(
    ("member", "wire_id"),
    [
        (CerebrasModel.GPT_OSS_120B, "gpt-oss-120b"),
        (CerebrasModel.GEMMA_4_31B, "gemma-4-31b"),
        (CerebrasModel.ZAI_GLM_4_7, "zai-glm-4.7"),
    ],
)
def test_cerebras_model_values(member, wire_id):
    assert member.value == wire_id


@pytest.mark.parametrize(
    ("member", "wire_value"),
    [
        (ReasoningEffort.NONE, "none"),
        (ReasoningEffort.LOW, "low"),
        (ReasoningEffort.MEDIUM, "medium"),
        (ReasoningEffort.HIGH, "high"),
    ],
)
def test_reasoning_effort_values(member, wire_value):
    assert member.value == wire_value


def test_every_model_has_a_spec():
    for model in CerebrasModel:
        assert model in CEREBRAS_MODEL_SPECS


@pytest.mark.parametrize(
    ("model", "efforts"),
    [
        (
            CerebrasModel.GPT_OSS_120B,
            (ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
        ),
        (
            CerebrasModel.GEMMA_4_31B,
            (ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
        ),
        (
            CerebrasModel.ZAI_GLM_4_7,
            (
                ReasoningEffort.NONE,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            ),
        ),
    ],
)
def test_reasoning_efforts_match_source(model, efforts):
    assert CEREBRAS_MODEL_SPECS[model].reasoning_efforts == efforts


@pytest.mark.parametrize("model", list(CerebrasModel))
def test_spec_token_budgets_and_reasoning_flag(model):
    spec = CEREBRAS_MODEL_SPECS[model]
    assert spec.context_tokens == 131072
    assert spec.response_tokens == 40000
    assert spec.reasoning is True


def test_spec_is_frozen():
    spec = CEREBRAS_MODEL_SPECS[CerebrasModel.GPT_OSS_120B]

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.context_tokens = 1
