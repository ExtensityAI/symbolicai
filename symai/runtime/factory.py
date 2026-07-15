from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import httpx
from pydantic import SecretStr

from symai.providers import cerebras, deepseek, openai
from symai.providers.cerebras.engines import chat_completions as cerebras_engine
from symai.providers.deepseek.engines import chat_completions as deepseek_engine
from symai.providers.openai.engines import embedding as openai_embedding
from symai.providers.openai.engines import responses as openai_responses
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.errors import EngineCapability, UnsupportedCapabilityError, UnsupportedModelError
from symai.runtime.models import NamedEngineConfig, Provider, RuntimeConfig, TransportConfig
from symai.runtime.runtime import Runtime

type _Capability = EngineCapability
type _ProviderEngine = LanguageModelEngine | EmbeddingEngine


@dataclass(frozen=True, slots=True)
class _EngineFactory:
    models: Mapping[str, object]
    create: Callable[[str, SecretStr, TransportConfig], _ProviderEngine]


@dataclass(frozen=True, slots=True)
class _ResolvedEngine:
    capability: _Capability
    config: NamedEngineConfig
    factory: _EngineFactory


def _timeout(config: TransportConfig) -> httpx.Timeout:
    return httpx.Timeout(
        config.request_timeout,
        connect=config.connect_timeout,
    )


def _create_openai_language_model(
    model: str,
    api_key: SecretStr,
    transport: TransportConfig,
) -> LanguageModelEngine:
    client = openai.Client(
        api_key=api_key,
        timeout=_timeout(transport),
        connect_retries=transport.connect_retries,
    )
    return openai.ResponsesEngine(client=client, model=model)


def _create_openai_embedding(
    model: str,
    api_key: SecretStr,
    transport: TransportConfig,
) -> EmbeddingEngine:
    client = openai.Client(
        api_key=api_key,
        timeout=_timeout(transport),
        connect_retries=transport.connect_retries,
    )
    return openai.EmbeddingEngine(client=client, model=model)


def _create_cerebras_language_model(
    model: str,
    api_key: SecretStr,
    transport: TransportConfig,
) -> LanguageModelEngine:
    client = cerebras.Client(
        api_key=api_key,
        timeout=_timeout(transport),
        connect_retries=transport.connect_retries,
    )
    return cerebras.ChatCompletionsEngine(client=client, model=model)


def _create_deepseek_language_model(
    model: str,
    api_key: SecretStr,
    transport: TransportConfig,
) -> LanguageModelEngine:
    client = deepseek.Client(
        api_key=api_key,
        timeout=_timeout(transport),
        connect_retries=transport.connect_retries,
    )
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
    constructed: list[_ProviderEngine] = []
    language_models: dict[str, LanguageModelEngine] = {}
    embeddings: dict[str, EmbeddingEngine] = {}

    try:
        for resolved_engine in resolved:
            engine = _create_engine(resolved_engine)
            constructed.append(engine)
            if resolved_engine.capability == "language_model":
                language_models[resolved_engine.config.name] = cast(
                    "LanguageModelEngine",
                    engine,
                )
            else:
                embeddings[resolved_engine.config.name] = cast("EmbeddingEngine", engine)

        return Runtime(
            language_models=language_models,
            embeddings=embeddings,
            default_language_model=config.default_language_model,
            default_embedding=config.default_embedding,
        )
    except BaseException as error:
        for engine in reversed(constructed):
            try:
                engine.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Runtime construction cleanup failed: {cleanup_error!r}",
                )
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


def _create_engine(resolved: _ResolvedEngine) -> _ProviderEngine:
    return resolved.factory.create(
        resolved.config.model,
        resolved.config.api_key,
        resolved.config.transport,
    )

