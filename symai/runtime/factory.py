from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import httpx
from pydantic import SecretStr

from symai.backend.engine_handle import EngineHandle
from symai.backend.engines.embedding import openai as openai_embedding
from symai.backend.engines.language_model import cerebras, deepseek, openai
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.openai.client import Client as OpenAIClient
from symai.runtime.errors import UnsupportedCapabilityError, UnsupportedModelError
from symai.runtime.models import Provider, ProviderEngineConfig, RuntimeConfig, TransportConfig
from symai.runtime.runtime import EmbeddingEngine, LanguageModelEngine, Runtime

_Capability = Literal["language_model", "embedding"]
_ProviderEngine = LanguageModelEngine | EmbeddingEngine


@dataclass(frozen=True, slots=True)
class _EngineFactory:
    models: Mapping[str, object]
    create: Callable[[str, SecretStr, httpx.Client], _ProviderEngine]


@dataclass(frozen=True, slots=True)
class _ResolvedEngine:
    capability: _Capability
    config: ProviderEngineConfig
    factory: _EngineFactory


def _create_openai_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = OpenAIClient(api_key=api_key, http_client=http_client)
    return openai.LanguageModelEngine(client=client, model=model)


def _create_openai_embedding(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> EmbeddingEngine:
    client = OpenAIClient(api_key=api_key, http_client=http_client)
    return openai_embedding.EmbeddingEngine(client=client, model=model)


def _create_cerebras_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = CerebrasClient(api_key=api_key, http_client=http_client)
    return cerebras.LanguageModelEngine(client=client, model=model)


def _create_deepseek_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = DeepSeekClient(api_key=api_key, http_client=http_client)
    return deepseek.LanguageModelEngine(client=client, model=model)


_FACTORIES: Mapping[tuple[Provider, _Capability], _EngineFactory] = MappingProxyType(
    {
        (Provider.OPENAI, "language_model"): _EngineFactory(
            models=openai.MODEL_SPECS,
            create=_create_openai_language_model,
        ),
        (Provider.OPENAI, "embedding"): _EngineFactory(
            models=openai_embedding.MODEL_SPECS,
            create=_create_openai_embedding,
        ),
        (Provider.CEREBRAS, "language_model"): _EngineFactory(
            models=cerebras.MODEL_SPECS,
            create=_create_cerebras_language_model,
        ),
        (Provider.DEEPSEEK, "language_model"): _EngineFactory(
            models=deepseek.MODEL_SPECS,
            create=_create_deepseek_language_model,
        ),
    }
)


def create_runtime(config: RuntimeConfig) -> Runtime:
    """Build a single-owner runtime after resolving every configured capability."""
    resolved = _resolve_config(config)
    language_model: EngineHandle[LanguageModelEngine] | None = None
    embedding: EngineHandle[EmbeddingEngine] | None = None
    constructed: list[EngineHandle[_ProviderEngine]] = []

    try:
        for engine in resolved:
            handle = _create_handle(engine)
            constructed.append(handle)
            if engine.capability == "language_model":
                language_model = cast("EngineHandle[LanguageModelEngine]", handle)
            else:
                embedding = cast("EngineHandle[EmbeddingEngine]", handle)

        return Runtime(language_model=language_model, embedding=embedding)
    except BaseException as error:
        _close_after_construction_failure(constructed, error)
        raise


def _resolve_config(config: RuntimeConfig) -> tuple[_ResolvedEngine, ...]:
    configured: list[tuple[_Capability, ProviderEngineConfig]] = []
    if config.language_model is not None:
        configured.append(("language_model", config.language_model))
    if config.embedding is not None:
        configured.append(("embedding", config.embedding))

    return tuple(_resolve_engine(capability, engine) for capability, engine in configured)


def _resolve_engine(
    capability: _Capability,
    config: ProviderEngineConfig,
) -> _ResolvedEngine:
    factory = _FACTORIES.get((config.provider, capability))
    if factory is None:
        msg = f"Provider {config.provider.value} does not support {capability}"
        raise UnsupportedCapabilityError(msg)
    if config.model not in factory.models:
        msg = f"Unsupported {config.provider.value} {capability} model: {config.model}"
        raise UnsupportedModelError(msg)
    return _ResolvedEngine(capability=capability, config=config, factory=factory)


def _create_handle(resolved: _ResolvedEngine) -> EngineHandle[_ProviderEngine]:
    http_client = _create_http_client(resolved.config.transport)
    try:
        engine = resolved.factory.create(
            resolved.config.model,
            resolved.config.api_key,
            http_client,
        )
    except BaseException as error:
        _close_with_note(http_client.close, error)
        raise

    return EngineHandle(engine=engine, cleanup=http_client.close)


def _create_http_client(config: TransportConfig) -> httpx.Client:
    timeout = httpx.Timeout(
        config.request_timeout,
        connect=config.connect_timeout,
    )
    transport = httpx.HTTPTransport(retries=config.connect_retries)
    try:
        return httpx.Client(timeout=timeout, transport=transport)
    except BaseException as error:
        _close_with_note(transport.close, error)
        raise


def _close_after_construction_failure(
    handles: Sequence[EngineHandle[_ProviderEngine]],
    error: BaseException,
) -> None:
    for handle in reversed(handles):
        _close_with_note(handle.close, error)


def _close_with_note(cleanup: Callable[[], None], error: BaseException) -> None:
    try:
        cleanup()
    except BaseException as cleanup_error:
        error.add_note(f"Runtime construction cleanup failed: {cleanup_error!r}")
