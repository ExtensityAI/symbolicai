from collections.abc import Callable, Mapping

from symai.providers._http.settings import HttpProviderSettings
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.errors import UnsupportedModelError


def load_responses(settings: Mapping[str, object]) -> Callable[[], LanguageModelEngine]:
    """Validate settings and return a factory that allocates the client.

    Resolution is separated from allocation so the runtime can validate every configured
    engine before any transport exists (FIXPLAN §2).
    """
    from symai.providers.openai.client.client import Client
    from symai.providers.openai.engines.responses import (
        MODEL_SPECS,
        UNSUPPORTED_MODEL_MESSAGE,
        ResponsesEngine,
    )

    parsed = HttpProviderSettings.model_validate(dict(settings))
    if parsed.model not in MODEL_SPECS:
        msg = UNSUPPORTED_MODEL_MESSAGE.format(model=parsed.model)
        raise UnsupportedModelError(msg)

    def construct() -> LanguageModelEngine:
        return ResponsesEngine(client=Client.from_settings(parsed), model=parsed.model)

    return construct


def load_embedding(settings: Mapping[str, object]) -> Callable[[], EmbeddingEngine]:
    """Validate settings and return a factory that allocates the client.

    Resolution is separated from allocation so the runtime can validate every configured
    engine before any transport exists (FIXPLAN §2).
    """
    from symai.providers.openai.client.client import Client
    from symai.providers.openai.engines.embedding import MODEL_SPECS, UNSUPPORTED_MODEL_MESSAGE
    from symai.providers.openai.engines.embedding import EmbeddingEngine as OpenAIEmbeddingEngine

    parsed = HttpProviderSettings.model_validate(dict(settings))
    if parsed.model not in MODEL_SPECS:
        msg = UNSUPPORTED_MODEL_MESSAGE.format(model=parsed.model)
        raise UnsupportedModelError(msg)

    def construct() -> EmbeddingEngine:
        return OpenAIEmbeddingEngine(client=Client.from_settings(parsed), model=parsed.model)

    return construct
