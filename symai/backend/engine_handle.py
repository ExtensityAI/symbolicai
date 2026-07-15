from collections.abc import Callable
from threading import Lock
from typing import Generic, Literal, TypeVar

EngineT_co = TypeVar("EngineT_co", covariant=True)
EngineCapability = Literal["language_model", "embedding"]


class EngineHandle(Generic[EngineT_co]):
    """Engine plus optional resource cleanup owned by its composition root."""

    __slots__ = ("_capability", "_cleanup", "_engine", "_lock", "_name")

    def __init__(
        self,
        name: str,
        capability: EngineCapability,
        engine: EngineT_co,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._name = name
        self._capability = capability
        self._engine = engine
        self._cleanup = cleanup
        self._lock = Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capability(self) -> EngineCapability:
        return self._capability

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
