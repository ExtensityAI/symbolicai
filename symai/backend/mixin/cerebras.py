from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CerebrasModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    reasoning_efforts: tuple[str, ...]


CEREBRAS_MODEL_SPECS = {
    "gpt-oss-120b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    "gemma-4-31b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    "zai-glm-4.7": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("none", "low", "medium", "high"),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_CEREBRAS_MODELS = [f"cerebras:{model}" for model in CEREBRAS_MODEL_SPECS]


class CerebrasMixin:
    model: str

    def cerebras_strip_prefix(self, model_name: str) -> str:
        if model_name.startswith("cerebras:"):
            return model_name.removeprefix("cerebras:")
        return model_name

    def cerebras_model_spec_for(self, model: str) -> CerebrasModelSpec:
        model_id = self.cerebras_strip_prefix(model)
        try:
            return CEREBRAS_MODEL_SPECS[model_id]
        except KeyError as e:
            msg = f"Unsupported Cerebras model: {model}"
            raise ValueError(msg) from e

    def cerebras_model_spec(self) -> CerebrasModelSpec:
        return self.cerebras_model_spec_for(self.model)

    def is_cerebras_reasoning_model(self) -> bool:
        return self.cerebras_model_spec().reasoning

    def api_max_context_tokens(self) -> int:
        return self.cerebras_model_spec().context_tokens

    def api_max_response_tokens(self) -> int:
        return self.cerebras_model_spec().response_tokens
