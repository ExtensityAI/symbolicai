"""Keyless search through Parallel's Streamable HTTP MCP endpoint.

Uses the shared HTTP transport, not a vendor SDK. MCP connection state is local
to a call; the research session belongs to the caller, never this cached engine.
"""

import json
import logging
from copy import deepcopy
from urllib.parse import urlsplit
from uuid import uuid4

from symai.backend.base import Engine
from symai.backend.engines.search.parallel.engine import (
    ParallelExtractResult,
    ParallelSearchResult,
)
from symai.backend.engines.search.parallel.models import (
    ParallelExtractResponse,
    ParallelSearchResponse,
)
from symai.backend.engines.search.parallel_mcp.models import (
    MCP_PROTOCOL,
    MCP_URL,
    USER_AGENT,
    FetchArguments,
    RPCRequest,
    SearchArguments,
)
from symai.backend.request import EngineAPIRequest, EngineRequestPayload
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import execute_engine_api_request, iter_sse_events

logger = logging.getLogger(__name__)


class ParallelMCPEngine(Engine):
    def __init__(self, *, client_timeout: float = 30):
        super().__init__(client_timeout=client_timeout)
        self.config = deepcopy(SYMAI_CONFIG)
        self.transport_client = None

    def id(self) -> str:
        model = self.config.get("SEARCH_ENGINE_MODEL")
        if self.config.get("SEARCH_ENGINE_ENABLED") is False:
            return super().id()
        if model == "parallel-mcp" or (
            (model is None or model == "") and not self.config.get("SEARCH_ENGINE_API_KEY")
        ):
            return "search"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        for key in ("SEARCH_ENGINE_MODEL", "SEARCH_ENGINE_API_KEY", "SEARCH_ENGINE_ENABLED"):
            if key in kwargs:
                self.config[key] = kwargs[key]

    def prepare(self, argument):
        # An ordinary decorated search is a standalone run unless the caller
        # supplies context. Interface("parallel") owns context across follow-ups.
        if not argument.prop.parallel_mcp_session_id:
            argument.prop.parallel_mcp_session_id = argument.kwargs.get("session_id") or str(
                uuid4()
            )
        argument.prop.prepared_input = argument.prop.query

    def build_request(self, argument) -> EngineAPIRequest:
        if self.id() != "search":
            msg = "Parallel MCP search is disabled or another search provider is configured."
            raise ValueError(msg)
        unsupported = set(argument.kwargs) & {
            "processor",
            "mode",
            "max_results",
            "max_chars_total",
            "max_chars_per_result",
            "allowed_domains",
            "excluded_domains",
            "source_policy",
            "advanced_settings",
        }
        if unsupported:
            msg = f"Parallel MCP does not support {sorted(unsupported)}; use the keyed parallel engine."
            raise ValueError(msg)
        kwargs = self.collect_request_kwargs(
            argument,
            {
                "objective",
                "search_queries",
                "session_id",
                "url",
                "full_content",
                "query",
            },
        )
        session_id = argument.prop.parallel_mcp_session_id
        url = kwargs.get("url")
        if url is not None:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                msg = "Parallel MCP fetch requires an HTTP or HTTPS URL."
                raise ValueError(msg)
            tool = "web_fetch"
            arguments = FetchArguments(
                urls=[url],
                objective=kwargs.get("objective"),
                search_queries=kwargs.get("search_queries"),
                full_content=kwargs.get("full_content", False),
                session_id=session_id,
            )
        else:
            query = argument.prop.prepared_input
            queries = kwargs.get("search_queries", [query] if isinstance(query, str) else query)
            if (
                not isinstance(queries, list)
                or not queries
                or any(not isinstance(q, str) or not q.strip() for q in queries)
            ):
                msg = "Parallel MCP search requires at least one non-empty query."
                raise ValueError(msg)
            tool = "web_search"
            arguments = SearchArguments(
                objective=kwargs.get("objective") or "\n".join(queries),
                search_queries=queries,
                session_id=session_id,
            )
        return self._request(
            "tools/call",
            {
                "name": tool,
                "arguments": arguments.model_dump(exclude_none=True),
            },
            timeout=self.client_timeout,
        )

    @staticmethod
    def _request(method, params=None, *, headers=None, timeout=30, notification=False):
        return EngineAPIRequest(
            provider="parallel-mcp",
            operation=method,
            payload=RPCRequest(
                id=None if notification else str(uuid4()),
                method=method,
                params=params,
            ),
            url=MCP_URL,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                **(headers or {}),
            },
            timeout=timeout,
        )

    def _send(self, request):
        # Do not automatically replay a tool call after a transport failure.
        response = execute_engine_api_request(request, client=self.transport_client, max_retries=0)
        if request.payload.id is None:
            return None, response.headers
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            messages = (json.loads(event.data) for event in iter_sse_events(response) if event.data)
        else:
            messages = iter([response.json()])
        for message in messages:
            if message.get("id") == request.payload.id:
                if "error" in message:
                    msg = f"Parallel MCP error: {message['error']}"
                    raise ValueError(msg)
                return message["result"], response.headers
        msg = "Parallel MCP returned no matching JSON-RPC response."
        raise ValueError(msg)

    def call_request(self, request):
        initialized, response_headers = self._send(
            self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "symbolicai", "version": "2.2.0"},
                },
                timeout=request.timeout,
            )
        )
        if initialized.get("protocolVersion") != MCP_PROTOCOL:
            msg = "Parallel MCP returned an unsupported protocol version."
            raise ValueError(msg)
        headers = {"MCP-Protocol-Version": MCP_PROTOCOL}
        if response_headers.get("mcp-session-id"):
            headers["Mcp-Session-Id"] = response_headers["mcp-session-id"]
        try:
            return self._call_tool(request, headers)
        finally:
            if "Mcp-Session-Id" in headers:
                try:
                    execute_engine_api_request(
                        EngineAPIRequest(
                            provider="parallel-mcp",
                            operation="close",
                            method="DELETE",
                            payload=EngineRequestPayload(),
                            url=MCP_URL,
                            headers={**request.headers, **headers},
                            timeout=request.timeout,
                        ),
                        client=self.transport_client,
                        max_retries=0,
                    )
                except Exception:
                    # Teardown cannot mask a tool error or cancellation. A server
                    # may also refuse explicit session termination with HTTP 405.
                    logger.debug("Could not terminate Parallel MCP connection", exc_info=True)

    def _call_tool(self, request, headers):
        self._send(
            self._request(
                "notifications/initialized",
                headers=headers,
                timeout=request.timeout,
                notification=True,
            )
        )
        # The server may evolve independently of the installed package.
        discovery, _ = self._send(
            self._request("tools/list", {}, headers=headers, timeout=request.timeout)
        )
        tool = request.payload.params["name"]
        if not any(t.get("name") == tool for t in discovery.get("tools", [])):
            msg = f"Parallel MCP does not advertise {tool}."
            raise ValueError(msg)
        result, _ = self._send(
            self._request(
                "tools/call", request.payload.params, headers=headers, timeout=request.timeout
            )
        )
        if result.get("isError"):
            msg = f"Parallel MCP tool failed: {result.get('content', [])}"
            raise ValueError(msg)
        content = result.get("structuredContent")
        if content is None:
            text = "\n".join(
                c["text"] for c in result.get("content", []) if c.get("type") == "text"
            )
            content = json.loads(text)
        response_type = ParallelSearchResponse if tool == "web_search" else ParallelExtractResponse
        return response_type.model_validate(content)

    @staticmethod
    def parse_response(response):
        if isinstance(response, ParallelSearchResponse):
            return [ParallelSearchResult(response)], {"raw_output": response}
        return [ParallelExtractResult(response)], {"raw_output": response}

    def forward(self, argument):
        return self.parse_response(self.call_request(self.build_request(argument)))
