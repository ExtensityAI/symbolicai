from dataclasses import dataclass
from enum import StrEnum


class CerebrasModel(StrEnum):
    GPT_OSS_120B = "gpt-oss-120b"
    GEMMA_4_31B = "gemma-4-31b"
    ZAI_GLM_4_7 = "zai-glm-4.7"


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class CerebrasModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    reasoning_efforts: tuple[ReasoningEffort, ...]


CEREBRAS_MODEL_SPECS: dict[CerebrasModel, CerebrasModelSpec] = {
    CerebrasModel.GPT_OSS_120B: CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
    ),
    CerebrasModel.GEMMA_4_31B: CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=(ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH),
    ),
    CerebrasModel.ZAI_GLM_4_7: CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=(
            ReasoningEffort.NONE,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        ),
    ),
}
