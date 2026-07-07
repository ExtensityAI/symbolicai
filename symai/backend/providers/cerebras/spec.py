from enum import StrEnum

from symai.backend.providers.models import StrictModel


class Model(StrEnum):
    GPT_OSS_120B = "gpt-oss-120b"
    GEMMA_4_31B = "gemma-4-31b"
    ZAI_GLM_4_7 = "zai-glm-4.7"


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningSpec(StrictModel):
    efforts: tuple[ReasoningEffort, ...]


class ModelSpec(StrictModel):
    context_tokens: int
    response_tokens: int
    # None => non-reasoning model; a ReasoningSpec => reasoning model with its allowed efforts.
    reasoning: ReasoningSpec | None


MODEL_SPECS: dict[Model, ModelSpec] = {
    Model.GPT_OSS_120B: ModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=ReasoningSpec(
            efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH)
        ),
    ),
    Model.GEMMA_4_31B: ModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=ReasoningSpec(
            efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH)
        ),
    ),
    Model.ZAI_GLM_4_7: ModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=ReasoningSpec(
            efforts=(
                ReasoningEffort.NONE,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        ),
    ),
}
