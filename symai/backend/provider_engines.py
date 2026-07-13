from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import cast

import httpx

from symai.backend.base import Engine
from symai.backend.engine_handle import EngineHandle
from symai.backend.engines.embedding import openai as openai_embedding_engine
from symai.backend.engines.language_model import cerebras as cerebras_engine
from symai.backend.engines.language_model import deepseek as deepseek_engine
from symai.backend.engines.language_model import openai as openai_language_engine
from symai.clients import cerebras, deepseek, openai


class Provider(StrEnum):
    OPENAI = "openai"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"


class Capability(StrEnum):
    LANGUAGE_MODEL = "language_model"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    provider: Provider
    capability: Capability
    model: str


@dataclass(frozen=True, slots=True)
class ProviderTransportOptions:
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


@dataclass(frozen=True, slots=True)
class _EngineFactory:
    models: Mapping[str, object]
    create: Callable[[str, str, httpx.Client], Engine]


def _create_openai_language_model(
    model: str,
    api_key: str,
    http_client: httpx.Client,
) -> Engine:
    client = openai.Client(api_key=api_key, http_client=http_client)
    return openai_language_engine.LanguageModelEngine(
        client=client,
        model=cast("openai.responses.ResponseModel", model),
    )


def _create_openai_embedding(
    model: str,
    api_key: str,
    http_client: httpx.Client,
) -> Engine:
    client = openai.Client(api_key=api_key, http_client=http_client)
    return openai_embedding_engine.EmbeddingEngine(
        client=client,
        model=cast("openai.embeddings.EmbeddingModel", model),
    )


def _create_cerebras_language_model(
    model: str,
    api_key: str,
    http_client: httpx.Client,
) -> Engine:
    client = cerebras.Client(api_key=api_key, http_client=http_client)
    return cerebras_engine.LanguageModelEngine(
        client=client,
        model=cast("cerebras.chat.ChatModel", model),
    )


def _create_deepseek_language_model(
    model: str,
    api_key: str,
    http_client: httpx.Client,
) -> Engine:
    client = deepseek.Client(api_key=api_key, http_client=http_client)
    return deepseek_engine.LanguageModelEngine(
        client=client,
        model=cast("deepseek.chat.ChatModel", model),
    )


_FACTORIES: Mapping[tuple[Provider, Capability], _EngineFactory] = {
    (Provider.OPENAI, Capability.LANGUAGE_MODEL): _EngineFactory(
        models=openai.responses.MODEL_SPECS,
        create=_create_openai_language_model,
    ),
    (Provider.OPENAI, Capability.EMBEDDING): _EngineFactory(
        models=openai.embeddings.MODEL_SPECS,
        create=_create_openai_embedding,
    ),
    (Provider.CEREBRAS, Capability.LANGUAGE_MODEL): _EngineFactory(
        models=cerebras.chat.MODEL_SPECS,
        create=_create_cerebras_language_model,
    ),
    (Provider.DEEPSEEK, Capability.LANGUAGE_MODEL): _EngineFactory(
        models=deepseek.chat.MODEL_SPECS,
        create=_create_deepseek_language_model,
    ),
}


def _parse_provider_selection(
    capability: Capability,
    model: str,
) -> ProviderSelection | None:
    provider_name, separator, model_id = model.partition(":")
    if not separator:
        if any(model in factory.models for factory in _FACTORIES.values()):
            msg = f"Managed provider model {model!r} must include a provider prefix"
            raise ValueError(msg)

        return None

    if not provider_name:
        msg = "provider must not be empty"
        raise ValueError(msg)

    if not model_id:
        msg = "model must not be empty"
        raise ValueError(msg)

    try:
        provider = Provider(provider_name)
    except ValueError as exc:
        msg = f"Unknown provider: {provider_name}"
        raise ValueError(msg) from exc

    return ProviderSelection(
        provider=provider,
        capability=capability,
        model=model_id,
    )


def create_provider_http_client(options: ProviderTransportOptions) -> httpx.Client:
    timeout = httpx.Timeout(
        options.request_timeout,
        connect=options.connect_timeout,
    )
    transport = httpx.HTTPTransport(retries=options.connect_retries)
    return httpx.Client(timeout=timeout, transport=transport)


def create_provider_engine_handle(
    *,
    capability: Capability,
    model: str,
    api_key: str,
    options: ProviderTransportOptions | None = None,
) -> EngineHandle | None:
    selection = _parse_provider_selection(capability, model)
    if selection is None:
        return None

    factory = _FACTORIES.get((selection.provider, selection.capability))
    if factory is None:
        msg = (
            f"Provider {selection.provider.value!r} does not support "
            f"capability {selection.capability.value!r}"
        )
        raise ValueError(msg)

    if selection.model not in factory.models:
        msg = (
            f"Unsupported {selection.provider.value} "
            f"{selection.capability.value} model: {selection.model}"
        )
        raise ValueError(msg)

    if not api_key:
        msg = "api_key must not be empty"
        raise ValueError(msg)

    http_client = create_provider_http_client(options or ProviderTransportOptions())
    try:
        engine = factory.create(selection.model, api_key, http_client)
    except BaseException:
        http_client.close()
        raise

    return EngineHandle(engine=engine, cleanup=http_client.close)
