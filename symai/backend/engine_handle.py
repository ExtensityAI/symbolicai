from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar

EngineT_co = TypeVar("EngineT_co", covariant=True)


class EngineHandle(Generic[EngineT_co]):
    """Engine plus optional resource cleanup owned by its composition root."""

    __slots__ = ("_cleanup", "_engine", "_lock")

    def __init__(
        self,
        engine: EngineT_co,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._engine = engine
        self._cleanup = cleanup
        self._lock = Lock()

    @property
    def engine(self) -> EngineT_co:
        return self._engine

    @property
    def owns_resources(self) -> bool:
        with self._lock:
            return self._cleanup is not None

    def close(self) -> None:
        with self._lock:
            cleanup = self._cleanup
            self._cleanup = None
        if cleanup is not None:
            cleanup()
