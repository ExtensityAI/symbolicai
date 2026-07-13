from typing import Literal

import httpx

from symai.backend.base import Engine
from symai.backend.engines.embedding.openai import (
    MODEL_SPECS as OPENAI_EMBEDDING_MODEL_SPECS,
)
from symai.backend.engines.embedding.openai import EmbeddingEngine as OpenAIEmbeddingAdapter
from symai.backend.engines.language_model.cerebras import (
    MODEL_SPECS as CEREBRAS_MODEL_SPECS,
)
from symai.backend.engines.language_model.cerebras import (
    LanguageModelEngine as CerebrasLanguageModelAdapter,
)
from symai.backend.engines.language_model.deepseek import (
    MODEL_SPECS as DEEPSEEK_MODEL_SPECS,
)
from symai.backend.engines.language_model.deepseek import (
    LanguageModelEngine as DeepSeekLanguageModelAdapter,
)
from symai.backend.engines.language_model.openai import (
    MODEL_SPECS as OPENAI_MODEL_SPECS,
)
from symai.backend.engines.language_model.openai import (
    LanguageModelEngine as OpenAILanguageModelAdapter,
)
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.openai.client import Client as OpenAIClient


def create_provider_http_client(
    *,
    capability: Literal["embedding", "language_model"],
    model: str,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> httpx.Client:
    if timeout is None and (capability == "embedding" or model.startswith("openai:")):
        client_timeout: httpx.Timeout | float | None = httpx.Timeout(600.0, connect=10.0)
    elif timeout is not None and (capability == "embedding" or model.startswith("openai:")):
        client_timeout = httpx.Timeout(timeout, connect=10.0)
    else:
        client_timeout = timeout

    transport = None
    if max_retries is not None:
        transport = httpx.HTTPTransport(retries=max_retries)
    return httpx.Client(timeout=client_timeout, transport=transport)


def create_provider_engine(
    *,
    capability: Literal["embedding", "language_model"],
    model: str,
    api_key: str,
    http_client: httpx.Client,
) -> Engine:
    if capability == "embedding":
        if model not in OPENAI_EMBEDDING_MODEL_SPECS:
            msg = f"Unsupported embedding model: {model}"
            raise ValueError(msg)
        return OpenAIEmbeddingAdapter(
            client=OpenAIClient(api_key=api_key, http_client=http_client),
            model=model,
        )

    if model.startswith("openai:"):
        model_id = model.removeprefix("openai:")
        if model_id not in OPENAI_MODEL_SPECS:
            msg = f"Unsupported OpenAI model: {model}"
            raise ValueError(msg)
        return OpenAILanguageModelAdapter(
            client=OpenAIClient(api_key=api_key, http_client=http_client),
            model=model_id,
        )

    if model.startswith("cerebras:"):
        model_id = model.removeprefix("cerebras:")
        if model_id not in CEREBRAS_MODEL_SPECS:
            msg = f"Unsupported Cerebras model: {model}"
            raise ValueError(msg)
        return CerebrasLanguageModelAdapter(
            client=CerebrasClient(api_key=api_key, http_client=http_client),
            model=model_id,
        )

    if model not in DEEPSEEK_MODEL_SPECS:
        msg = f"Unsupported language model: {model}"
        raise ValueError(msg)
    return DeepSeekLanguageModelAdapter(
        client=DeepSeekClient(api_key=api_key, http_client=http_client),
        model=model,
    )
