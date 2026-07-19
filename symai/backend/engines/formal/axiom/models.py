"""Axiom (axle) API wire models.

Locked against https://axle.axiommath.ai — POST /api/v1/{tool}, NDJSON single-line
response, Bearer auth. Derived from axiom-axle 1.4.0 (client.py/types.py).
"""

# NOTE: no `from __future__ import annotations` — pydantic resolves annotations at
# class-creation time, so JsonValue must stay a runtime import (TC002 false positive).
from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

AXLE_API_BASE = "https://axle.axiommath.ai"
AXLE_TOOL_URL = f"{AXLE_API_BASE}/api/v1"

DEFAULT_ENVIRONMENT = "lean-4.28.0"


class AxiomPayload(EngineRequestPayload):
    """Fields every tool request shares; tool-specific fields ride in extra_body."""

    content: str | list[str] | None = None
    environment: str = DEFAULT_ENVIRONMENT
    ignore_imports: bool = True
    timeout_seconds: JsonValue | None = None


class AxiomResponse(EngineResponsePayload):
    """Tolerant per-tool response; the server answers 200 with error keys on failure."""

    okay: bool = False
    content: JsonValue | None = None
    lean_messages: JsonValue | None = None
    tool_messages: JsonValue | None = None
    failed_declarations: list[str] = Field(default_factory=list)
    timings: dict[str, JsonValue] | None = None
    info: JsonValue | None = None
    internal_error: str | None = None
    user_error: str | None = None
    error: str | None = None
