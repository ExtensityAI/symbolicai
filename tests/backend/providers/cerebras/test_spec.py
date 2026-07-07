import pytest
from pydantic import ValidationError

from symai.backend.providers.cerebras.spec import (
    MODEL_SPECS,
    Model,
    ReasoningEffort,
    ReasoningSpec,
)


@pytest.mark.parametrize(
    ("member", "wire_id"),
    [
        (Model.GPT_OSS_120B, "gpt-oss-120b"),
        (Model.GEMMA_4_31B, "gemma-4-31b"),
        (Model.ZAI_GLM_4_7, "zai-glm-4.7"),
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
    for model in Model:
        assert model in MODEL_SPECS


@pytest.mark.parametrize(
    ("model", "efforts"),
    [
        (
            Model.GPT_OSS_120B,
            (ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
        ),
        (
            Model.GEMMA_4_31B,
            (ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
        ),
        (
            Model.ZAI_GLM_4_7,
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
    reasoning = MODEL_SPECS[model].reasoning

    assert reasoning is not None
    assert reasoning.efforts == efforts


@pytest.mark.parametrize("model", list(Model))
def test_spec_token_budgets(model):
    spec = MODEL_SPECS[model]
    assert spec.context_tokens == 131072
    assert spec.response_tokens == 40000


def test_spec_is_frozen():
    spec = MODEL_SPECS[Model.GPT_OSS_120B]

    with pytest.raises(ValidationError):
        spec.context_tokens = 1


def test_reasoning_spec_is_frozen():
    reasoning = ReasoningSpec(efforts=(ReasoningEffort.LOW,))

    with pytest.raises(ValidationError):
        reasoning.efforts = (ReasoningEffort.HIGH,)
