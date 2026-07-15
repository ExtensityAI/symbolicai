from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import httpx
from pydantic import SecretStr

from symai.backend.engine_handle import EngineCapability, EngineHandle
from symai.providers import cerebras, deepseek, openai
from symai.providers.cerebras.engines import chat_completions as cerebras_engine
from symai.providers.deepseek.engines import chat_completions as deepseek_engine
from symai.providers.openai.engines import embedding as openai_embedding
from symai.providers.openai.engines import responses as openai_responses
from symai.runtime.errors import UnsupportedCapabilityError, UnsupportedModelError
from symai.runtime.models import NamedEngineConfig, Provider, RuntimeConfig, TransportConfig
from symai.runtime.runtime import EmbeddingEngine, LanguageModelEngine, Runtime

_Capability: TypeAlias = EngineCapability
_ProviderEngine = LanguageModelEngine | EmbeddingEngine


@dataclass(frozen=True, slots=True)
class _EngineFactory:
    models: Mapping[str, object]
    create: Callable[[str, SecretStr, httpx.Client], _ProviderEngine]


@dataclass(frozen=True, slots=True)
class _ResolvedEngine:
    capability: _Capability
    config: NamedEngineConfig
    factory: _EngineFactory


def _create_openai_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = openai.Client(api_key=api_key, http_client=http_client)
    return openai.ResponsesEngine(client=client, model=model)


def _create_openai_embedding(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> EmbeddingEngine:
    client = openai.Client(api_key=api_key, http_client=http_client)
    return openai.EmbeddingEngine(client=client, model=model)


def _create_cerebras_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = cerebras.Client(api_key=api_key, http_client=http_client)
    return cerebras.ChatCompletionsEngine(client=client, model=model)


def _create_deepseek_language_model(
    model: str,
    api_key: SecretStr,
    http_client: httpx.Client,
) -> LanguageModelEngine:
    client = deepseek.Client(api_key=api_key, http_client=http_client)
    return deepseek.ChatCompletionsEngine(client=client, model=model)


_FACTORIES: Mapping[tuple[Provider, _Capability], _EngineFactory] = MappingProxyType(
    {
        (Provider.OPENAI, "language_model"): _EngineFactory(
            models=openai_responses.MODEL_SPECS,
            create=_create_openai_language_model,
        ),
        (Provider.OPENAI, "embedding"): _EngineFactory(
            models=openai_embedding.MODEL_SPECS,
            create=_create_openai_embedding,
        ),
        (Provider.CEREBRAS, "language_model"): _EngineFactory(
            models=cerebras_engine.MODEL_SPECS,
            create=_create_cerebras_language_model,
        ),
        (Provider.DEEPSEEK, "language_model"): _EngineFactory(
            models=deepseek_engine.MODEL_SPECS,
            create=_create_deepseek_language_model,
        ),
    }
)


def create_runtime(config: RuntimeConfig) -> Runtime:
    """Build a single-owner runtime after resolving every configured instance."""
    resolved = _resolve_config(config)
    constructed: list[EngineHandle[_ProviderEngine]] = []

    try:
        for engine in resolved:
            constructed.append(_create_handle(engine))

        return Runtime._from_engine_handles(
            constructed,
            default_language_model=config.default_language_model,
            default_embedding=config.default_embedding,
        )
    except BaseException as error:
        _close_after_construction_failure(constructed, error)
        raise


def _resolve_config(config: RuntimeConfig) -> tuple[_ResolvedEngine, ...]:
    return (
        *(_resolve_engine("language_model", engine) for engine in config.language_models),
        *(_resolve_engine("embedding", engine) for engine in config.embeddings),
    )


def _resolve_engine(
    capability: _Capability,
    config: NamedEngineConfig,
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

    return EngineHandle(
        name=resolved.config.name,
        capability=resolved.capability,
        engine=engine,
        cleanup=http_client.close,
    )


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
