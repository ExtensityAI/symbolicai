from collections.abc import Callable, Mapping

from symai.providers._client.settings import HttpProviderSettings
from symai.runtime.engines import LanguageModelEngine
from symai.runtime.errors import UnsupportedModelError


def load_chat_completions(settings: Mapping[str, object]) -> Callable[[], LanguageModelEngine]:
    """Validate settings and return a factory that allocates the client.

    Resolution is separated from allocation so the runtime can validate every configured
    engine before any transport exists.
    """

    parsed = HttpProviderSettings.model_validate(dict(settings))

    from symai.providers.cerebras.engines.chat_completions import (
        MODEL_SPECS,
        UNSUPPORTED_MODEL_MESSAGE,
        ChatCompletionsEngine,
    )

    if parsed.model not in MODEL_SPECS:
        msg = UNSUPPORTED_MODEL_MESSAGE.format(model=parsed.model)
        raise UnsupportedModelError(msg)

    def construct() -> LanguageModelEngine:
        import httpx

        from symai.providers.cerebras.client import Client

        client = Client(
            api_key=parsed.api_key,
            timeout=httpx.Timeout(
                parsed.request_timeout,
                connect=parsed.connect_timeout,
            ),
            connect_retries=parsed.connect_retries,
        )
        return ChatCompletionsEngine(client=client, model=parsed.model)

    return construct
