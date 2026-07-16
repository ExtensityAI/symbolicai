from collections.abc import Callable, Mapping

from symai.providers._engine.loading import resolve_http_engine
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine


def load_responses(settings: Mapping[str, object]) -> Callable[[], LanguageModelEngine]:
    """Validate settings and return a factory that allocates the client."""
    from symai.providers.openai.client import Client
    from symai.providers.openai.engines.responses import (
        MODEL_SPECS,
        UNSUPPORTED_MODEL_MESSAGE,
        ResponsesEngine,
    )

    return resolve_http_engine(
        settings,
        model_specs=MODEL_SPECS,
        unsupported_model_message=UNSUPPORTED_MODEL_MESSAGE,
        client=Client,
        engine=ResponsesEngine,
    )


def load_embedding(settings: Mapping[str, object]) -> Callable[[], EmbeddingEngine]:
    """Validate settings and return a factory that allocates the client."""
    from symai.providers.openai.client import Client
    from symai.providers.openai.engines.embedding import MODEL_SPECS, UNSUPPORTED_MODEL_MESSAGE
    from symai.providers.openai.engines.embedding import EmbeddingEngine as OpenAIEmbeddingEngine

    return resolve_http_engine(
        settings,
        model_specs=MODEL_SPECS,
        unsupported_model_message=UNSUPPORTED_MODEL_MESSAGE,
        client=Client,
        engine=OpenAIEmbeddingEngine,
    )
