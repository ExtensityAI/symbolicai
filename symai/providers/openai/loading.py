from collections.abc import Mapping

from symai.providers.openai.settings import EmbeddingSettings, ResponsesSettings
from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.errors import UnsupportedModelError


def load_responses(settings: Mapping[str, object]) -> LanguageModelEngine:
    parsed = ResponsesSettings.model_validate(dict(settings))

    import httpx

    from symai.providers.openai.client import Client
    from symai.providers.openai.engines.responses import MODEL_SPECS, ResponsesEngine

    if parsed.model not in MODEL_SPECS:
        msg = f"Unsupported OpenAI language model: {parsed.model}"
        raise UnsupportedModelError(msg)

    client = Client(
        api_key=parsed.api_key,
        timeout=httpx.Timeout(
            parsed.request_timeout,
            connect=parsed.connect_timeout,
        ),
        connect_retries=parsed.connect_retries,
    )
    return ResponsesEngine(client=client, model=parsed.model)


def load_embedding(settings: Mapping[str, object]) -> EmbeddingEngine:
    parsed = EmbeddingSettings.model_validate(dict(settings))

    import httpx

    from symai.providers.openai.client import Client
    from symai.providers.openai.engines.embedding import MODEL_SPECS
    from symai.providers.openai.engines.embedding import EmbeddingEngine as OpenAIEmbeddingEngine

    if parsed.model not in MODEL_SPECS:
        msg = f"Unsupported OpenAI embedding model: {parsed.model}"
        raise UnsupportedModelError(msg)

    client = Client(
        api_key=parsed.api_key,
        timeout=httpx.Timeout(
            parsed.request_timeout,
            connect=parsed.connect_timeout,
        ),
        connect_retries=parsed.connect_retries,
    )
    return OpenAIEmbeddingEngine(client=client, model=parsed.model)
