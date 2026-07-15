from typing import Protocol

from symai.runtime.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    LanguageModelRequest,
    LanguageModelResponse,
)


class LanguageModelEngine(Protocol):
    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse: ...

    def close(self) -> None: ...


class EmbeddingEngine(Protocol):
    def execute(self, request: EmbeddingRequest, /) -> EmbeddingResponse: ...

    def close(self) -> None: ...
