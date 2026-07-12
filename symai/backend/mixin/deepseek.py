from __future__ import annotations

from dataclasses import dataclass

# https://api-docs.deepseek.com/quick_start/pricing


@dataclass(frozen=True)
class DeepSeekModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool


DEEPSEEK_MODEL_SPECS = {
    "deepseek-v4-flash": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
    ),
    "deepseek-v4-pro": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
    ),
}

SUPPORTED_MODELS = list(DEEPSEEK_MODEL_SPECS)


class DeepSeekMixin:
    model: str

    def deepseek_model_spec_for(self, model: str) -> DeepSeekModelSpec:
        try:
            return DEEPSEEK_MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported DeepSeek model: {model}"
            raise ValueError(msg) from e

    def deepseek_model_spec(self) -> DeepSeekModelSpec:
        return self.deepseek_model_spec_for(self.model)

    def api_max_context_tokens(self) -> int:
        return self.deepseek_model_spec().context_tokens
