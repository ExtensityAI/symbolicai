from __future__ import annotations

from contextvars import ContextVar, Token
from enum import StrEnum
from threading import Condition, Lock
from types import MappingProxyType
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
    from collections.abc import Mapping, Sequence
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

    _condition: Condition
    _construction_order: tuple[EngineHandle[object], ...]
    _default_embedding: str | None
    _default_language_model: str | None
    _embedding: EngineHandle[EmbeddingEngine] | None
    _engine_handles: Mapping[str, EngineHandle[object]]
    _in_flight: int
    _language_model: EngineHandle[LanguageModelEngine] | None
    _state: _RuntimeState
    _token: Token[Runtime | None] | None

    __slots__ = (
        "_condition",
        "_construction_order",
        "_default_embedding",
        "_default_language_model",
        "_embedding",
        "_engine_handles",
        "_in_flight",
        "_language_model",
        "_state",
        "_token",
    )

    def __init__(self) -> None:
        msg = "Runtime instances must be created with create_runtime()"
        raise TypeError(msg)

    @classmethod
    def _from_engine_handles(
        cls,
        handles: Sequence[EngineHandle[object]],
        *,
        default_language_model: str | None,
        default_embedding: str | None,
    ) -> Runtime:
        construction_order = tuple(handles)
        language_model = cls._resolve_default_handle(
            construction_order,
            "language_model",
            default_language_model,
        )
        embedding = cls._resolve_default_handle(
            construction_order,
            "embedding",
            default_embedding,
        )
        runtime = cls.__new__(cls)
        runtime._condition = Condition(Lock())
        runtime._language_model = cast(
            "EngineHandle[LanguageModelEngine] | None",
            language_model,
        )
        runtime._embedding = cast("EngineHandle[EmbeddingEngine] | None", embedding)
        runtime._construction_order = construction_order
        runtime._engine_handles = MappingProxyType(
            {handle.name: handle for handle in construction_order}
        )
        runtime._default_language_model = default_language_model
        runtime._default_embedding = default_embedding
        runtime._in_flight = 0
        runtime._state = _RuntimeState.CREATED
        runtime._token = None
        return runtime

    @staticmethod
    def _resolve_default_handle(
        handles: Sequence[EngineHandle[object]],
        capability: Literal["language_model", "embedding"],
        default: str | None,
    ) -> EngineHandle[object] | None:
        if default is not None:
            return next(
                handle
                for handle in handles
                if handle.name == default and handle.capability == capability
            )
        return next(
            (handle for handle in handles if handle.capability == capability),
            None,
        )

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
        handles = tuple(reversed(self._construction_order))
        self._construction_order = ()
        self._engine_handles = MappingProxyType({})
        self._language_model = None
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
