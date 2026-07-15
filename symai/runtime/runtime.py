from __future__ import annotations

from contextvars import ContextVar, Token
from enum import StrEnum
from threading import Lock, get_ident
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

from symai.runtime.errors import (
    AmbiguousEngineError,
    EngineCapability,
    EngineCapabilityError,
    NoActiveRuntimeError,
    RuntimeClosedError,
    RuntimeOwnershipError,
    UnknownEngineError,
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

    from symai.backend import engine_handle


class LanguageModelEngine(Protocol):
    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse: ...


class EmbeddingEngine(Protocol):
    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class _RuntimeState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"


class Runtime:
    """Explicit context-scoped owner for normalized execution capabilities."""

    _construction_order: tuple[engine_handle.EngineHandle[object], ...]
    _defaults: Mapping[EngineCapability, str | None]
    _embeddings: Mapping[str, engine_handle.EngineHandle[EmbeddingEngine]]
    _engine_handles: Mapping[str, engine_handle.EngineHandle[object]]
    _language_models: Mapping[str, engine_handle.EngineHandle[LanguageModelEngine]]
    _lifecycle_lock: Lock
    _owner_thread_id: int | None
    _state: _RuntimeState
    _token: Token[Runtime | None] | None

    __slots__ = (
        "_construction_order",
        "_defaults",
        "_embeddings",
        "_engine_handles",
        "_language_models",
        "_lifecycle_lock",
        "_owner_thread_id",
        "_state",
        "_token",
    )

    def __init__(self) -> None:
        msg = "Runtime instances must be created with create_runtime()"
        raise TypeError(msg)

    @classmethod
    def _from_engine_handles(
        cls,
        handles: Sequence[engine_handle.EngineHandle[object]],
        *,
        default_language_model: str | None,
        default_embedding: str | None,
    ) -> Runtime:
        construction_order = tuple(handles)
        engine_handles: dict[str, engine_handle.EngineHandle[object]] = {}
        language_models: dict[str, engine_handle.EngineHandle[LanguageModelEngine]] = {}
        embeddings: dict[str, engine_handle.EngineHandle[EmbeddingEngine]] = {}

        for handle in construction_order:
            if handle.name in engine_handles:
                msg = f"Duplicate engine name: {handle.name!r}"
                raise ValueError(msg)

            engine_handles[handle.name] = handle
            if handle.capability == "language_model":
                language_models[handle.name] = cast(
                    "engine_handle.EngineHandle[LanguageModelEngine]",
                    handle,
                )
            else:
                embeddings[handle.name] = cast(
                    "engine_handle.EngineHandle[EmbeddingEngine]",
                    handle,
                )

        cls._validate_default(
            "language_model",
            default_language_model,
            language_models,
        )
        cls._validate_default("embedding", default_embedding, embeddings)

        runtime = cls.__new__(cls)
        runtime._construction_order = construction_order
        runtime._engine_handles = MappingProxyType(engine_handles)
        runtime._language_models = MappingProxyType(language_models)
        runtime._embeddings = MappingProxyType(embeddings)
        runtime._defaults = MappingProxyType(
            {
                "language_model": default_language_model,
                "embedding": default_embedding,
            }
        )
        runtime._lifecycle_lock = Lock()
        runtime._owner_thread_id = None
        runtime._state = _RuntimeState.CREATED
        runtime._token = None
        return runtime

    @staticmethod
    def _validate_default(
        capability: EngineCapability,
        default: str | None,
        handles: Mapping[str, engine_handle.EngineHandle[object]],
    ) -> None:
        if default is None or default in handles:
            return

        msg = f"Default {capability} engine is not configured: {default!r}"
        raise ValueError(msg)

    def __enter__(self) -> Runtime:
        with self._lifecycle_lock:
            self._require_owner_thread("enter")
            if self._state is not _RuntimeState.CREATED:
                msg = "Runtime contexts have a single lifecycle and cannot be re-entered"
                raise RuntimeClosedError(msg)

            token = _CURRENT_RUNTIME.set(self)
            self._token = token
            self._owner_thread_id = get_ident()
            self._state = _RuntimeState.ACTIVE
            return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        with self._lifecycle_lock:
            self._require_owner_thread("exit")
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
    def execute(
        self,
        request: LanguageModelRequest,
        *,
        engine: str | None = None,
    ) -> LanguageModelResponse: ...

    @overload
    def execute(
        self,
        request: EmbeddingRequest,
        *,
        engine: str | None = None,
    ) -> EmbeddingResponse: ...

    def execute(
        self,
        request: LanguageModelRequest | EmbeddingRequest,
        *,
        engine: str | None = None,
    ) -> LanguageModelResponse | EmbeddingResponse:
        with self._lifecycle_lock:
            self._require_owner_thread("execute")
            if self._state is not _RuntimeState.ACTIVE:
                msg = "Runtime only accepts execution while its context is active"
                raise RuntimeClosedError(msg)

            if isinstance(request, LanguageModelRequest):
                handle = self._resolve_engine(
                    "language_model",
                    self._language_models,
                    engine,
                )
            elif isinstance(request, EmbeddingRequest):
                handle = self._resolve_engine("embedding", self._embeddings, engine)
            else:
                msg = f"Unsupported runtime request type: {type(request).__name__}"
                raise TypeError(msg)

        if isinstance(request, LanguageModelRequest):
            language_engine = cast("LanguageModelEngine", handle.engine)
            return language_engine.execute(request)
        embedding_engine = cast("EmbeddingEngine", handle.engine)
        return embedding_engine.execute(request)

    def _resolve_engine(
        self,
        capability: EngineCapability,
        handles: Mapping[str, engine_handle.EngineHandle[object]],
        engine_name: str | None,
    ) -> engine_handle.EngineHandle[object]:
        if engine_name is not None:
            selected = self._engine_handles.get(engine_name)
            if selected is None:
                raise UnknownEngineError(engine_name)
            if selected.capability != capability:
                raise EngineCapabilityError(
                    engine_name,
                    requested_capability=capability,
                    engine_capability=selected.capability,
                )
            return selected

        default = self._defaults[capability]
        if default is not None:
            return handles[default]
        if len(handles) == 1:
            return next(iter(handles.values()))
        if handles:
            raise AmbiguousEngineError(capability, tuple(handles))

        capability_label = capability.replace("_", "-")
        msg = f"Runtime has no {capability_label} capability"
        raise UnsupportedCapabilityError(msg)

    def close(self) -> None:
        with self._lifecycle_lock:
            self._require_owner_thread("close")
            if self._state is _RuntimeState.CLOSED:
                return

            self._state = _RuntimeState.CLOSED
            handles = self._detach_handles()

        failures: list[BaseException] = []
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                failures.append(error)

        if failures:
            msg = "Runtime cleanup failed"
            raise BaseExceptionGroup(msg, failures)

    def _require_owner_thread(
        self,
        operation: Literal["enter", "execute", "close", "exit"],
    ) -> None:
        owner_thread_id = self._owner_thread_id
        if owner_thread_id is None or owner_thread_id == get_ident():
            return

        raise RuntimeOwnershipError(operation)

    def _detach_handles(self) -> tuple[engine_handle.EngineHandle[object], ...]:
        handles = tuple(reversed(self._construction_order))
        self._construction_order = ()
        self._engine_handles = MappingProxyType({})
        self._language_models = MappingProxyType({})
        self._embeddings = MappingProxyType({})
        self._defaults = MappingProxyType({})
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
