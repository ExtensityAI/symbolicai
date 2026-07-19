import logging
from copy import deepcopy
from typing import ClassVar

from symai.backend.base import Engine
from symai.backend.engines.formal.axiom.models import (
    AXLE_TOOL_URL,
    AxiomPayload,
    AxiomResponse,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import execute_engine_api_request
from symai.symbol import Result

logger = logging.getLogger(__name__)


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

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = self.config.get("FORMAL_ENGINE_API_KEY") if api_key is None else api_key
        self.name = self.__class__.__name__
        self.transport_client = None

    def id(self) -> str:
        if self.config.get("FORMAL_ENGINE_API_KEY") and self.config.get("FORMAL_ENGINE") == "axiom":
            return "formal"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "FORMAL_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["FORMAL_ENGINE_API_KEY"]

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def build_request(self, argument) -> EngineAPIRequest:
        tool = argument.kwargs.get("tool", "check")
        config = dict(argument.kwargs.get("config", {}))

        if tool not in self.TOOLS:
            msg = f"Unknown tool '{tool}'. Must be one of: {', '.join(sorted(self.TOOLS))}"
            raise ValueError(msg)

        content = argument.prop.prepared_input
        # default environment to latest Lean 4 if not specified
        config.setdefault("environment", "lean-4.28.0")
        # ignore import mismatches by default for standalone snippets
        config.setdefault("ignore_imports", True)

        if tool == "merge":
            # merge takes `documents` (list of strings) instead of `content`
            config["documents"] = content if isinstance(content, list) else [content]
            content = None
        elif tool == "verify_proof":
            # verify_proof takes `formal_statement` as a separate required arg
            config["formal_statement"] = config.pop("formal_statement")

        payload = AxiomPayload(
            content=content,
            environment=config.pop("environment"),
            ignore_imports=config.pop("ignore_imports"),
            timeout_seconds=config.pop("timeout_seconds", None),
        )
        # NOTE: the server speaks NDJSON (exactly one JSON line), so the timeout is a
        # server-side budget; the transport client stays unbounded like other engines.
        return EngineAPIRequest(
            provider="axiom",
            operation=tool,
            payload=payload,
            url=f"{AXLE_TOOL_URL}/{tool}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            extra_body=config,
        )

    def call_request(self, request: EngineAPIRequest) -> AxiomResponse:
        response = execute_engine_api_request(request, client=self.transport_client)
        # NDJSON: exactly one JSON line per request
        lines = [line for line in response.text.strip().split("\n") if line]
        if len(lines) != 1:
            msg = f"Expected 1 response line from Axiom, got {len(lines)}"
            raise RuntimeError(msg)
        return AxiomResponse.model_validate_json(lines[0])

    def parse_response(self, response: AxiomResponse) -> tuple[list[Result], dict]:
        if response.internal_error:
            raise RuntimeError(response.internal_error)
        if response.user_error:
            raise ValueError(response.user_error)
        if response.error:
            raise RuntimeError(response.error)
        raw = response.model_dump(exclude={"internal_error", "user_error", "error"})
        return [AxiomResult(raw)], {"raw_output": response}

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)
