from collections.abc import Callable, Mapping

from symai.providers._engine.loading import resolve_http_engine
from symai.runtime.engines import LanguageModelEngine


def load_chat_completions(settings: Mapping[str, object]) -> Callable[[], LanguageModelEngine]:
    """Validate settings and return a factory that allocates the client."""
    from symai.providers.deepseek.client import Client
    from symai.providers.deepseek.engines.chat_completions import (
        MODEL_SPECS,
        UNSUPPORTED_MODEL_MESSAGE,
        ChatCompletionsEngine,
    )

    return resolve_http_engine(
        settings,
        model_specs=MODEL_SPECS,
        unsupported_model_message=UNSUPPORTED_MODEL_MESSAGE,
        client=Client,
        engine=ChatCompletionsEngine,
    )
