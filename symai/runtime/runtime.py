from __future__ import annotations

from builtins import BaseExceptionGroup
from contextvars import ContextVar, Token
from enum import StrEnum
from threading import Condition, Lock
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

from symai.runtime.errors import (
    NoActiveRuntimeError,
    RuntimeClosedError,
    UnsupportedCapabilityError,
)
from symai.runtime.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    LanguageModelRequest,
    LanguageModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from symai.backend.engine_handle import EngineHandle


class LanguageModelEngine(Protocol):
    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse: ...


class EmbeddingEngine(Protocol):
    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class _RuntimeState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


class Runtime:
    """Explicit context-scoped owner for normalized execution capabilities."""

    __slots__ = (
        "_condition",
        "_embedding",
        "_in_flight",
        "_language_model",
        "_state",
        "_token",
    )

    def __init__(
        self,
        *,
        language_model: EngineHandle[LanguageModelEngine] | None = None,
        embedding: EngineHandle[EmbeddingEngine] | None = None,
    ) -> None:
        self._condition = Condition(Lock())
        self._language_model = language_model
        self._embedding = embedding
        self._in_flight = 0
        self._state = _RuntimeState.CREATED
        self._token: Token[Runtime | None] | None = None

    def __enter__(self) -> Runtime:
        with self._condition:
            if self._state is not _RuntimeState.CREATED:
                msg = "Runtime contexts have a single lifecycle and cannot be re-entered"
                raise RuntimeClosedError(msg)

            token = _CURRENT_RUNTIME.set(self)
            self._token = token
            self._state = _RuntimeState.ACTIVE

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        token = self._token
        if token is None:
            msg = "Active runtime context is missing its context token"
            raise RuntimeError(msg)

        _CURRENT_RUNTIME.reset(token)
        self._token = None

        try:
            self.close()
        except BaseExceptionGroup as cleanup_failures:
            if exc_value is None:
                raise
            for failure in cleanup_failures.exceptions:
                exc_value.add_note(f"Runtime cleanup failed: {failure!r}")

        return False

    @overload
    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse: ...

    @overload
    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    def execute(
        self,
        request: LanguageModelRequest | EmbeddingRequest,
    ) -> LanguageModelResponse | EmbeddingResponse:
        with self._condition:
            if self._state is not _RuntimeState.ACTIVE:
                msg = "Runtime only accepts execution while its context is active"
                raise RuntimeClosedError(msg)

            if isinstance(request, LanguageModelRequest):
                handle = self._language_model
                if handle is None:
                    msg = "Runtime has no language-model capability"
                    raise UnsupportedCapabilityError(msg)
            elif isinstance(request, EmbeddingRequest):
                handle = self._embedding
                if handle is None:
                    msg = "Runtime has no embedding capability"
                    raise UnsupportedCapabilityError(msg)
            else:
                msg = f"Unsupported runtime request type: {type(request).__name__}"
                raise TypeError(msg)

            self._in_flight += 1

        try:
            if isinstance(request, LanguageModelRequest):
                engine = cast("LanguageModelEngine", handle.engine)
                return engine.execute(request)
            engine = cast("EmbeddingEngine", handle.engine)
            return engine.execute(request)
        finally:
            with self._condition:
                self._in_flight -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._state is _RuntimeState.CLOSED:
                return

            if self._state is _RuntimeState.CLOSING:
                self._condition.wait_for(lambda: self._state is _RuntimeState.CLOSED)
                return

            self._state = _RuntimeState.CLOSING
            self._condition.wait_for(lambda: self._in_flight == 0)
            handles = self._detach_handles()

        failures: list[BaseException] = []
        try:
            for handle in handles:
                try:
                    handle.close()
                except BaseException as error:
                    failures.append(error)
        finally:
            with self._condition:
                self._state = _RuntimeState.CLOSED
                self._condition.notify_all()

        if failures:
            msg = "Runtime cleanup failed"
            raise BaseExceptionGroup(msg, failures)

    def _detach_handles(self) -> Sequence[EngineHandle[object]]:
        handles: list[EngineHandle[object]] = []
        if self._language_model is not None:
            handles.append(self._language_model)
            self._language_model = None
        if self._embedding is not None:
            handles.append(self._embedding)
            self._embedding = None
        return handles


_CURRENT_RUNTIME: ContextVar[Runtime | None] = ContextVar(
    "symai_active_runtime",
    default=None,
)


def current_runtime() -> Runtime:
    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        msg = "No Runtime is active in the current context"
        raise NoActiveRuntimeError(msg)
    return runtime
