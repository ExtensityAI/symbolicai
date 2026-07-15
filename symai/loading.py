from collections.abc import Mapping, Sequence
from typing import cast

from symai.runtime.config import ImplementationId, RuntimeConfig
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.loading import (
    EmbeddingLoaderEntry,
    LanguageModelLoaderEntry,
    load_runtime as _load_runtime,
)
from symai.runtime.runtime import Runtime


def _load_openai_responses(settings: Mapping[str, object]) -> LanguageModelEngine:
    from symai.providers.openai.loading import load_responses

    return load_responses(settings)


def _load_openai_embedding(settings: Mapping[str, object]) -> EmbeddingEngine:
    from symai.providers.openai.loading import load_embedding

    return load_embedding(settings)


def _load_cerebras_chat_completions(
    settings: Mapping[str, object],
) -> LanguageModelEngine:
    from symai.providers.cerebras.loading import load_chat_completions

    return load_chat_completions(settings)


def _load_deepseek_chat_completions(
    settings: Mapping[str, object],
) -> LanguageModelEngine:
    from symai.providers.deepseek.loading import load_chat_completions

    return load_chat_completions(settings)


BUILTIN_LANGUAGE_MODEL_LOADERS: tuple[LanguageModelLoaderEntry, ...] = (
    (cast("ImplementationId", "openai:responses"), _load_openai_responses),
    (
        cast("ImplementationId", "cerebras:chat-completions"),
        _load_cerebras_chat_completions,
    ),
    (
        cast("ImplementationId", "deepseek:chat-completions"),
        _load_deepseek_chat_completions,
    ),
)
BUILTIN_EMBEDDING_LOADERS: tuple[EmbeddingLoaderEntry, ...] = (
    (cast("ImplementationId", "openai:embeddings"), _load_openai_embedding),
)


def load_runtime(
    config: RuntimeConfig,
    *,
    language_model_loaders: Sequence[LanguageModelLoaderEntry] = (),
    embedding_loaders: Sequence[EmbeddingLoaderEntry] = (),
) -> Runtime:
    """Compose immutable built-ins with explicit extension entries and load a Runtime."""
    return _load_runtime(
        config,
        language_model_loaders=(
            *BUILTIN_LANGUAGE_MODEL_LOADERS,
            *language_model_loaders,
        ),
        embedding_loaders=(*BUILTIN_EMBEDDING_LOADERS, *embedding_loaders),
    )
