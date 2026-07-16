from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symai.providers.openai.client import Client as Client
    from symai.providers.openai.engines import (
        EmbeddingEngine as EmbeddingEngine,
    )
    from symai.providers.openai.engines import (
        ResponsesEngine as ResponsesEngine,
    )


def __getattr__(name: str) -> object:
    if name == "Client":
        from symai.providers.openai.client import Client

        globals()[name] = Client
        return Client
    if name == "ResponsesEngine":
        from symai.providers.openai.engines import ResponsesEngine

        globals()[name] = ResponsesEngine
        return ResponsesEngine
    if name == "EmbeddingEngine":
        from symai.providers.openai.engines import EmbeddingEngine

        globals()[name] = EmbeddingEngine
        return EmbeddingEngine

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
