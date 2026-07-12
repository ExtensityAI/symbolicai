from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool = True
    pro: bool = False
    tokenizer: str = "o200k_base"


OPENAI_MODEL_SPECS = {
    "gpt-5.5": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.5-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        pro=True,
    ),
    "gpt-5.4": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.4-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        pro=True,
    ),
    "gpt-5.4-mini": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.4-nano": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "o3-pro": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
        pro=True,
    ),
    "o3": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
    ),
    "gpt-4.1": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
    ),
    "gpt-4.1-mini": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
    ),
}

SUPPORTED_CHAT_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if not spec.reasoning]
SUPPORTED_REASONING_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if spec.reasoning]
SUPPORTED_OPENAI_MODELS = [f"openai:{model}" for model in OPENAI_MODEL_SPECS]


class OpenAIMixin:
    model: str

    def openai_model_spec_for(self, model: str) -> OpenAIModelSpec:
        try:
            return OPENAI_MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported model: {model}"
            raise ValueError(msg) from e

    def openai_model_spec(self) -> OpenAIModelSpec:
        return self.openai_model_spec_for(self.model)

    def is_openai_reasoning_model(self) -> bool:
        return self.openai_model_spec().reasoning

    def is_openai_pro_model(self) -> bool:
        return self.openai_model_spec().pro

    def supports_openai_vision(self) -> bool:
        return self.openai_model_spec().vision

    def openai_tokenizer_name(self) -> str:
        return self.openai_model_spec().tokenizer

    def api_max_context_tokens(self):
        return self.openai_model_spec().context_tokens

    def api_max_response_tokens(self):
        return self.openai_model_spec().response_tokens

    def api_embedding_context_tokens(self) -> int:
        if self.model.startswith("text-embedding"):
            return 8_191
        msg = f"Unsupported model: {self.model}"
        raise ValueError(msg)

    def api_embedding_dims(self):
        if self.model == "text-embedding-ada-002":
            return 1_536
        if self.model == "text-embedding-3-small":
            return 1_536
        if self.model == "text-embedding-3-large":
            return 3_072
        msg = f"Unsupported model: {self.model}"
        raise ValueError(msg)
