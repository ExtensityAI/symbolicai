"""Wolfram Alpha v2/query wire models.

Locked against https://products.wolframalpha.com/api/documentation (JSON output).
"""

# NOTE: no `from __future__ import annotations` — pydantic resolves annotations at
# class-creation time, so JsonValue must stay a runtime import (TC002 false positive).
from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

WOLFRAM_API_BASE = "https://api.wolframalpha.com"
WOLFRAM_QUERY_URL = f"{WOLFRAM_API_BASE}/v2/query"


class WolframQueryParams(EngineRequestPayload):
    """Query parameters for GET /v2/query (the API is GET-only; no JSON body)."""

    input: str
    appid: str
    output: str = "json"


class WolframSubpod(EngineResponsePayload):
    plaintext: str = ""
    img: JsonValue | None = None


class WolframPod(EngineResponsePayload):
    title: str = ""
    primary: bool = False
    subpods: list[WolframSubpod] = Field(default_factory=list)


class WolframQueryResult(EngineResponsePayload):
    success: bool
    error: bool | JsonValue = False
    numpods: int = 0
    pods: list[WolframPod] = Field(default_factory=list)
    didyoumeans: JsonValue | None = None


class WolframResponse(EngineResponsePayload):
    queryresult: WolframQueryResult
