from enum import StrEnum

from symai.backend.providers.cerebras.base import StrictModel


class CerebrasModel(StrEnum):
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


class CerebrasModelSpec(StrictModel):
    context_tokens: int
    response_tokens: int
    # None => non-reasoning model; a ReasoningSpec => reasoning model with its allowed efforts.
    reasoning: ReasoningSpec | None


CEREBRAS_MODEL_SPECS: dict[CerebrasModel, CerebrasModelSpec] = {
    CerebrasModel.GPT_OSS_120B: CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=ReasoningSpec(
            efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH)
        ),
    ),
    CerebrasModel.GEMMA_4_31B: CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=ReasoningSpec(
            efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH)
        ),
    ),
    CerebrasModel.ZAI_GLM_4_7: CerebrasModelSpec(
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
