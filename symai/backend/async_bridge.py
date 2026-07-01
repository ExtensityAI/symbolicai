import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def run_async(coro):
    """Run a coroutine on a persistent background event loop and return its result.

    Safe to call from any thread, including one whose own event loop is already
    running (e.g. a Jupyter kernel): the coroutine executes on a dedicated
    background loop, so no nested-loop patching (``nest_asyncio``) is required.
    """
    global _loop, _thread

    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, daemon=True)
            _thread.start()

    return asyncio.run_coroutine_threadsafe(coro, _loop).result()
