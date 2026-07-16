from collections.abc import Callable, Mapping, Sequence

from pydantic import TypeAdapter

from symai.runtime.config import EngineConfig, ImplementationId, RuntimeConfig
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.observability import Observer
from symai.runtime.runtime import Runtime

# Loading is two-phase: a loader resolves settings and returns a factory, and only the
# factory allocates a transport. This is what lets every configuration be validated
# before any HTTP client exists.
LanguageModelFactory = Callable[[], LanguageModelEngine]
EmbeddingFactory = Callable[[], EmbeddingEngine]
LanguageModelLoader = Callable[[Mapping[str, object]], LanguageModelFactory]
EmbeddingLoader = Callable[[Mapping[str, object]], EmbeddingFactory]
LanguageModelLoaderEntry = tuple[ImplementationId, LanguageModelLoader]
EmbeddingLoaderEntry = tuple[ImplementationId, EmbeddingLoader]

_ProviderEngine = LanguageModelEngine | EmbeddingEngine
_IMPLEMENTATION_ID_ADAPTER = TypeAdapter(ImplementationId)


def load_runtime(
    config: RuntimeConfig,
    *,
    language_model_loaders: Sequence[LanguageModelLoaderEntry],
    embedding_loaders: Sequence[EmbeddingLoaderEntry],
    observers: Sequence[Observer] = (),
) -> Runtime:
    """Load an immutable envelope after a complete allocation-free preflight."""
    language_index, embedding_index = _preflight(
        config,
        language_model_loaders,
        embedding_loaders,
    )
    # Resolve every configuration first. A settings error in the last engine must not
    # leave the first engine's HTTP client allocated.
    language_factories = _resolve("language model", config.language_models, language_index)
    embedding_factories = _resolve("embedding", config.embeddings, embedding_index)

    loaded: list[_ProviderEngine] = []
    language_models: dict[str, LanguageModelEngine] = {}
    embeddings: dict[str, EmbeddingEngine] = {}

    try:
        for alias, language_factory in language_factories.items():
            engine = language_factory()
            loaded.append(engine)
            language_models[alias] = engine
        for alias, embedding_factory in embedding_factories.items():
            embedding = embedding_factory()
            loaded.append(embedding)
            embeddings[alias] = embedding

        return Runtime(
            language_models=language_models,
            embeddings=embeddings,
            observers=observers,
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


def _resolve[FactoryT](
    operation: str,
    configured: Mapping[str, EngineConfig],
    loaders: Mapping[str, Callable[[Mapping[str, object]], FactoryT]],
) -> dict[str, FactoryT]:
    """Resolve each configuration to an engine factory without allocating transport."""
    factories: dict[str, FactoryT] = {}
    for alias, spec in configured.items():
        factory = loaders[spec.implementation](spec.settings)
        if not callable(factory):
            msg = (
                f"{operation.capitalize()} loader for {spec.implementation!r} must return "
                f"an engine factory (engine {alias!r})"
            )
            raise TypeError(msg)

        factories[alias] = factory

    return factories


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
    configured: Mapping[str, EngineConfig],
    loaders: Mapping[str, LoaderT],
) -> None:
    for alias, spec in configured.items():
        implementation = spec.implementation
        if implementation in loaders:
            continue

        msg = f"No {operation} loader for implementation {implementation!r} (engine {alias!r})"
        raise ValueError(msg)
