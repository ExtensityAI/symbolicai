from __future__ import annotations

from symai.backend.engines.neurosymbolic.deepseek.engine import DeepseekEngine
from symai.backend.engines.neurosymbolic.deepseek.models import (
    DeepSeekModelSpec,
    DeepSeekOptions,
    DeepSeekPayload,
    DeepSeekRequest,
)
from symai.backend.usage import ModelPricing
from symai.prompts import strip_cache_breakpoints_from_messages

API_PINNED = "2026-09-10"
ATLAS_CHAT_COMPLETIONS_URL = "https://api.atlascloud.ai/v1/chat/completions"

ATLAS_MODEL_SPECS = {
    "deepseek-ai/deepseek-v4-flash": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
        pricing=ModelPricing(input=0.14, output=0.28, cached_input=0.028),
    ),
    "deepseek-ai/deepseek-v4-pro": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
        pricing=ModelPricing(input=1.68, output=3.38, cached_input=0.13),
    ),
}

SUPPORTED_ATLAS_MODELS = [f"atlas:{model}" for model in ATLAS_MODEL_SPECS]


def atlas_strip_prefix(model_name: str) -> str:
    return model_name.removeprefix("atlas:")


def atlas_model_spec_for(model: str) -> DeepSeekModelSpec:
    model_id = atlas_strip_prefix(model)
    try:
        return ATLAS_MODEL_SPECS[model_id]
    except KeyError as exc:
        message = f"Unsupported Atlas Cloud model: {model}"
        raise ValueError(message) from exc


class AtlasEngine(DeepseekEngine):
    """Atlas Cloud chat-completions engine using its OpenAI-compatible API."""

    def id(self) -> str:
        if self.model in SUPPORTED_ATLAS_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()

    def api_max_context_tokens(self) -> int:
        return atlas_model_spec_for(self.model).context_tokens

    def build_request(self, argument) -> DeepSeekRequest:
        allowed_request_kwargs = set(DeepSeekPayload.model_fields).union(
            DeepSeekOptions.model_fields
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in DeepSeekOptions.model_fields
            if key in payload_kwargs
        }
        payload_kwargs["model"] = atlas_strip_prefix(payload_kwargs.get("model", self.model))
        atlas_model_spec_for(payload_kwargs["model"])
        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = "<|endoftext|>"
        if payload_kwargs.get("stream"):
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = DeepSeekPayload.model_validate(payload_kwargs)
        options = DeepSeekOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(request_options.get("extra_headers", {}))

        return DeepSeekRequest(
            provider="atlas",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=ATLAS_CHAT_COMPLETIONS_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )
