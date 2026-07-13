from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

import httpx

from symai.backend.engine_handle import EngineHandle
from symai.backend.engines.embedding.openai import EmbeddingEngine
from symai.backend.engines.language_model.cerebras import (
    LanguageModelEngine as CerebrasLanguageModelEngine,
)
from symai.backend.engines.language_model.deepseek import (
    LanguageModelEngine as DeepSeekLanguageModelEngine,
)
from symai.backend.engines.language_model.openai import (
    LanguageModelEngine as OpenAILanguageModelEngine,
)
from symai.clients.cerebras import chat as cerebras_chat
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.deepseek import chat as deepseek_chat
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.openai import embeddings as openai_embeddings
from symai.clients.openai import responses as openai_responses
from symai.clients.openai.client import Client as OpenAIClient

_MANAGED_PROVIDERS = frozenset({"openai", "cerebras", "deepseek"})
_MODEL_CATALOGS = (
    openai_responses.MODEL_SPECS,
    openai_embeddings.MODEL_SPECS,
    cerebras_chat.MODEL_SPECS,
    deepseek_chat.MODEL_SPECS,
)


@dataclass(frozen=True, slots=True)
class ProviderRuntimeOptions:
    request_timeout: float = 600.0
    connect_timeout: float = 10.0
    connect_retries: int = 0

    def __post_init__(self):
        if self.request_timeout <= 0 or not isfinite(self.request_timeout):
            msg = "request_timeout must be finite and greater than zero"
            raise ValueError(msg)
        if self.connect_timeout <= 0 or not isfinite(self.connect_timeout):
            msg = "connect_timeout must be finite and greater than zero"
            raise ValueError(msg)
        if self.connect_retries < 0:
            msg = "connect_retries must be non-negative"
            raise ValueError(msg)


def _known_model(model: str, catalog: Mapping[str, object], label: str) -> str:
    if model not in catalog:
        msg = f"Unsupported {label}: {model}"
        raise ValueError(msg)
    return model


def _parse_provider_model(model: str) -> tuple[str, str] | None:
    provider, separator, model_id = model.partition(":")
    if separator:
        return (provider, model_id) if provider in _MANAGED_PROVIDERS else None

    if any(model in catalog for catalog in _MODEL_CATALOGS):
        msg = f"Managed provider model {model!r} must include a provider prefix"
        raise ValueError(msg)
    return None


def create_provider_http_client(options: ProviderRuntimeOptions) -> httpx.Client:
    timeout = httpx.Timeout(options.request_timeout, connect=options.connect_timeout)
    transport = httpx.HTTPTransport(retries=options.connect_retries)
    return httpx.Client(timeout=timeout, transport=transport)


def create_provider_engine_handle(
    *,
    capability: Literal["embedding", "language_model"],
    model: str,
    api_key: str,
    options: ProviderRuntimeOptions | None = None,
) -> EngineHandle | None:
    provider_model = _parse_provider_model(model)
    if provider_model is None:
        return None
    provider, model_id = provider_model

    if not api_key:
        msg = "api_key must not be empty"
        raise ValueError(msg)
    if capability not in ("embedding", "language_model"):
        msg = f"Unsupported provider capability: {capability}"
        raise ValueError(msg)

    if capability == "embedding":
        if provider != "openai":
            msg = f"Provider {provider!r} does not provide embedding through this runtime"
            raise ValueError(msg)
        known_model = _known_model(
            model_id,
            openai_embeddings.MODEL_SPECS,
            "OpenAI embedding model",
        )
    elif provider == "openai":
        known_model = _known_model(
            model_id,
            openai_responses.MODEL_SPECS,
            "OpenAI response model",
        )
    elif provider == "cerebras":
        known_model = _known_model(
            model_id,
            cerebras_chat.MODEL_SPECS,
            "Cerebras chat model",
        )
    else:
        known_model = _known_model(
            model_id,
            deepseek_chat.MODEL_SPECS,
            "DeepSeek chat model",
        )

    http_client = create_provider_http_client(options or ProviderRuntimeOptions())
    try:
        if capability == "embedding":
            client = OpenAIClient(api_key=api_key, http_client=http_client)
            engine = EmbeddingEngine(
                client=client,
                model=cast("openai_embeddings.Model", known_model),
            )
        elif provider == "openai":
            client = OpenAIClient(api_key=api_key, http_client=http_client)
            engine = OpenAILanguageModelEngine(
                client=client,
                model=cast("openai_responses.Model", known_model),
            )
        elif provider == "cerebras":
            client = CerebrasClient(api_key=api_key, http_client=http_client)
            engine = CerebrasLanguageModelEngine(
                client=client,
                model=cast("cerebras_chat.Model", known_model),
            )
        else:
            client = DeepSeekClient(api_key=api_key, http_client=http_client)
            engine = DeepSeekLanguageModelEngine(
                client=client,
                model=cast("deepseek_chat.Model", known_model),
            )
    except BaseException:
        http_client.close()
        raise
    return EngineHandle(engine=engine, cleanup=http_client.close)
