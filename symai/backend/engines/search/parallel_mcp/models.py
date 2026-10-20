"""Parallel Search MCP schemas, verified against tools/list and the MCP docs."""

from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload

API_PINNED = "2026-10-20"
MCP_URL = "https://search.parallel.ai/mcp"
MCP_PROTOCOL = "2025-03-26"
USER_AGENT = "SymbolicAI/2.2.0 (https://github.com/ExtensityAI/symbolicai)"


class SearchArguments(EngineRequestPayload):
    objective: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)
    session_id: str = Field(min_length=1, max_length=100)


class FetchArguments(EngineRequestPayload):
    urls: list[str] = Field(min_length=1, max_length=20)
    objective: str | None = Field(default=None, max_length=200)
    search_queries: list[str] | None = None
    full_content: bool = False
    session_id: str = Field(min_length=1, max_length=100)


class RPCRequest(EngineRequestPayload):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | None = None
    method: str
    params: dict[str, JsonValue] | None = None
