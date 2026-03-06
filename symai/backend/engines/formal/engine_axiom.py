import asyncio
import dataclasses
import threading
from copy import deepcopy
from typing import ClassVar

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

try:
    from axle import AxleClient
except ImportError:
    AxleClient = None


class AxiomResult(Result):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.raw = value
        self._value = value


class AxiomEngine(Engine):
    TOOLS: ClassVar[set[str]] = {
        "verify_proof",
        "check",
        "extract_theorems",
        "rename",
        "theorem2lemma",
        "theorem2sorry",
        "merge",
        "simplify_theorems",
        "repair_proofs",
        "have2lemma",
        "have2sorry",
        "sorry2lemma",
        "disprove",
        "normalize",
    }
    _loop = None
    _thread = None
    _lock = threading.Lock()

    def __init__(self, api_key: str | None = None):
        super().__init__()
        if AxleClient is None:
            UserMessage(
                "axle is not installed. Install with `pip install symbolicai[lean]`",
                raise_with=ImportError,
            )
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = self.config.get("FORMAL_ENGINE_API_KEY") if api_key is None else api_key
        self.client = AxleClient(api_key=self.api_key)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get("FORMAL_ENGINE_API_KEY") and self.config.get("FORMAL_ENGINE") == "axiom":
            return "formal"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "FORMAL_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["FORMAL_ENGINE_API_KEY"]
            self.client = AxleClient(api_key=self.api_key)

    @classmethod
    def _ensure_loop(cls):
        """Start a persistent background event loop (once) shared across all calls."""
        with cls._lock:
            if cls._loop is None or cls._loop.is_closed():
                cls._loop = asyncio.new_event_loop()
                cls._thread = threading.Thread(target=cls._loop.run_forever, daemon=True)
                cls._thread.start()

    def _run_async(self, coro):
        """Submit a coroutine to the persistent background loop and wait for the result."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def forward(self, argument):
        tool = argument.kwargs.get("tool", "check")
        config = argument.kwargs.get("config", {})

        if tool not in self.TOOLS:
            UserMessage(
                f"Unknown tool '{tool}'. Must be one of: {', '.join(sorted(self.TOOLS))}",
                raise_with=ValueError,
            )

        fn = getattr(self.client, tool)
        content = argument.prop.prepared_input
        # default environment to latest Lean 4 if not specified
        config.setdefault("environment", "lean-4.28.0")
        # ignore import mismatches by default for standalone snippets
        config.setdefault("ignore_imports", True)

        if tool == "merge":
            # merge takes `documents` (list of strings) instead of `content`
            coro = fn(documents=content if isinstance(content, list) else [content], **config)
        elif tool == "verify_proof":
            # verify_proof takes `formal_statement` as a separate required arg
            formal_statement = config.pop("formal_statement")
            coro = fn(formal_statement=formal_statement, content=content, **config)
        else:
            coro = fn(content=content, **config)

        rsp = self._run_async(coro)

        # Convert dataclass response to dict for Symbol compatibility
        raw = dataclasses.asdict(rsp) if dataclasses.is_dataclass(rsp) else rsp
        return [AxiomResult(raw)], {}

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
