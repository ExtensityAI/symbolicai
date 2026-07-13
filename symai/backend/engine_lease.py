from collections.abc import Callable
from threading import Lock

from symai.backend.base import Engine


class EngineLease:
    """Engine plus optional resource cleanup owned by its composition root."""

    __slots__ = ("_cleanup", "_lock", "engine")

    def __init__(self, engine: Engine, cleanup: Callable[[], None] | None = None) -> None:
        self.engine = engine
        self._cleanup = cleanup
        self._lock = Lock()

    def _has_cleanup(self) -> bool:
        with self._lock:
            return self._cleanup is not None

    def close(self) -> None:
        with self._lock:
            cleanup = self._cleanup
            self._cleanup = None
        if cleanup is not None:
            cleanup()
