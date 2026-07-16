from collections.abc import Callable, Mapping

from symai.providers._client.settings import HttpProviderSettings
from symai.runtime.engines import LanguageModelEngine
from symai.runtime.errors import UnsupportedModelError


def load_chat_completions(settings: Mapping[str, object]) -> Callable[[], LanguageModelEngine]:
    """Validate settings and return a factory that allocates the client.

    Resolution is separated from allocation so the runtime can validate every configured
    engine before any transport exists (FIXPLAN §2).
    """
    from symai.providers.deepseek.client.client import Client
    from symai.providers.deepseek.engines.chat_completions import (
        MODEL_SPECS,
        UNSUPPORTED_MODEL_MESSAGE,
        ChatCompletionsEngine,
    )

    parsed = HttpProviderSettings.model_validate(dict(settings))
    if parsed.model not in MODEL_SPECS:
        msg = UNSUPPORTED_MODEL_MESSAGE.format(model=parsed.model)
        raise UnsupportedModelError(msg)

    def construct() -> LanguageModelEngine:
        return ChatCompletionsEngine(client=Client.from_settings(parsed), model=parsed.model)

    return construct
