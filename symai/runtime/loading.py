from collections.abc import Callable, Mapping, Sequence
from typing import cast

from pydantic import TypeAdapter

from symai.runtime.config import EngineSpec, ImplementationId, RuntimeConfig
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.runtime import Runtime

LanguageModelLoader = Callable[[Mapping[str, object]], LanguageModelEngine]
EmbeddingLoader = Callable[[Mapping[str, object]], EmbeddingEngine]
LanguageModelLoaderEntry = tuple[ImplementationId, LanguageModelLoader]
EmbeddingLoaderEntry = tuple[ImplementationId, EmbeddingLoader]

_ProviderEngine = LanguageModelEngine | EmbeddingEngine
_IMPLEMENTATION_ID_ADAPTER = TypeAdapter(ImplementationId)


def load_runtime(
    config: RuntimeConfig,
    *,
    language_model_loaders: Sequence[LanguageModelLoaderEntry],
    embedding_loaders: Sequence[EmbeddingLoaderEntry],
) -> Runtime:
    """Load an immutable envelope after a complete allocation-free preflight."""
    language_index, embedding_index = _preflight(
        config,
        language_model_loaders,
        embedding_loaders,
    )
    loaded: list[_ProviderEngine] = []
    language_models: dict[str, LanguageModelEngine] = {}
    embeddings: dict[str, EmbeddingEngine] = {}

    try:
        for alias, spec in config.language_models.items():
            engine = language_index[spec.implementation](spec.settings)
            loaded.append(engine)
            language_models[alias] = engine
        for alias, spec in config.embeddings.items():
            engine = embedding_index[spec.implementation](spec.settings)
            loaded.append(engine)
            embeddings[alias] = engine

        return Runtime(
            language_models=language_models,
            embeddings=embeddings,
            default_language_model=config.default_language_model,
            default_embedding=config.default_embedding,
        )
    except BaseException as error:
        cleanup_failures: list[BaseException] = []
        for engine in reversed(loaded):
            try:
                engine.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)

        if cleanup_failures:
            group = BaseExceptionGroup(
                "Runtime loading cleanup failed",
                cleanup_failures,
            )
            raise error from group
        raise


def _preflight(
    config: RuntimeConfig,
    language_model_loaders: Sequence[LanguageModelLoaderEntry],
    embedding_loaders: Sequence[EmbeddingLoaderEntry],
) -> tuple[dict[str, LanguageModelLoader], dict[str, EmbeddingLoader]]:
    language_index = _index_entries(language_model_loaders)
    embedding_index = _index_entries(embedding_loaders)
    _validate_references("language model", config.language_models, language_index)
    _validate_references("embedding", config.embeddings, embedding_index)
    return language_index, embedding_index


def _index_entries[LoaderT](
    entries: Sequence[tuple[ImplementationId, LoaderT]],
) -> dict[str, LoaderT]:
    index: dict[str, LoaderT] = {}
    for raw_implementation, loader in entries:
        implementation = _IMPLEMENTATION_ID_ADAPTER.validate_python(raw_implementation)
        if implementation in index:
            msg = f"Duplicate implementation ID: {implementation}"
            raise ValueError(msg)
        if not callable(loader):
            msg = f"Loader for {implementation} must be callable"
            raise TypeError(msg)

        index[implementation] = loader

    return index


def _validate_references[LoaderT](
    operation: str,
    configured: Mapping[str, EngineSpec],
    loaders: Mapping[str, LoaderT],
) -> None:
    for alias, spec in configured.items():
        implementation = cast("str", spec.implementation)
        if implementation in loaders:
            continue

        msg = (
            f"No {operation} loader for implementation {implementation!r} "
            f"(engine {alias!r})"
        )
        raise ValueError(msg)
