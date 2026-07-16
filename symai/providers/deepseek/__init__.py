from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symai.providers.deepseek.client import Client as Client
    from symai.providers.deepseek.engines import (
        ChatCompletionsEngine as ChatCompletionsEngine,
    )


def __getattr__(name: str) -> object:
    if name == "Client":
        from symai.providers.deepseek.client import Client

        globals()[name] = Client
        return Client
    if name == "ChatCompletionsEngine":
        from symai.providers.deepseek.engines import ChatCompletionsEngine

        globals()[name] = ChatCompletionsEngine
        return ChatCompletionsEngine

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
