"""Exercise MCP through the public interface and normal engine resolver."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import httpx
import pytest

from symai import EngineRepository, Interface, Symbol, core
from symai.backend.base import ENGINE_UNREGISTERED
from symai.backend.engines.search.parallel import ParallelEngine
from symai.backend.engines.search.parallel_mcp import ParallelMCPEngine
from symai.backend.engines.search.parallel_mcp.models import MCP_PROTOCOL, MCP_URL, USER_AGENT
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import EngineRateLimitError, EngineTransportError
from tests.engines.search.interface import assert_citation_contract


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setattr(EngineRepository(), "_engines", {})
    monkeypatch.setattr(EngineRepository(), "get_dynamic_engine_instance", lambda: None)
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", {})
    monkeypatch.setattr("symai.backend.engines.search.parallel.engine.SYMAI_CONFIG", {})
    monkeypatch.setattr("symai.backend.engines.search.firecrawl.engine.SYMAI_CONFIG", {})
    monkeypatch.setattr("symai.backend.engines.search.gemini.engine.SYMAI_CONFIG", {})
    monkeypatch.setattr("symai.backend.engines.search.openai.engine.SYMAI_CONFIG", {})
    monkeypatch.setattr("symai.backend.engines.search.perplexity.engine.SYMAI_CONFIG", {})
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_ENGINE_API_KEY", raising=False)


class MCPServer:
    def __init__(self, *, sse=False, text=False, failure=None):
        self.requests = []
        self.calls = []
        self.sse = sse
        self.text = text
        self.failure = failure

    def __call__(self, request):
        assert str(request.url) == MCP_URL
        assert request.headers["user-agent"] == USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        self.requests.append(request)
        if request.method == "DELETE":
            assert request.headers["mcp-session-id"]
            return httpx.Response(200)
        body = json.loads(request.content)
        method = body["method"]
        if method == "initialize":
            assert body["params"]["clientInfo"]["name"] == "symbolicai"
            result = {"protocolVersion": MCP_PROTOCOL, "capabilities": {}}
        else:
            assert request.headers["mcp-protocol-version"] == MCP_PROTOCOL
            assert request.headers["mcp-session-id"]
            if method == "notifications/initialized":
                assert "id" not in body
                return httpx.Response(202)
            if method == "tools/list":
                result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
            else:
                assert method == "tools/call"
                self.calls.append(body["params"])
                if self.failure:
                    return self.failure(request, body)
                args = body["params"]["arguments"]
                data = {
                    "session_id": args["session_id"],
                    "results": [
                        {
                            "url": "https://example.com/research?utm_source=search",
                            "title": "Research source",
                            "excerpts": ["Useful current research evidence."],
                        }
                    ],
                }
                if body["params"]["name"] == "web_search":
                    data["search_id"] = "search_fixture"
                else:
                    data["extract_id"] = "extract_fixture"
                    data["errors"] = []
                    if args.get("full_content"):
                        data["results"][0]["full_content"] = "Complete page markdown."
                result = {"content": [{"type": "text", "text": json.dumps(data)}]}
                if not self.text:
                    result["structuredContent"] = data
        message = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        headers = {"mcp-session-id": "connection-" + body["id"]}
        if self.sse:
            headers["content-type"] = "text/event-stream"
            return httpx.Response(
                200,
                headers=headers,
                text=": keepalive\n\nevent: message\ndata: " + json.dumps(message) + "\n\n",
            )
        return httpx.Response(200, headers=headers, json=message)


@pytest.fixture
def caller(monkeypatch):
    server = MCPServer()
    client = httpx.Client(transport=httpx.MockTransport(server))
    monkeypatch.setattr("symai.backend.transport.ENGINE_API_CLIENT", client)
    yield server
    client.close()


@pytest.mark.parametrize("sse,text", [(False, False), (True, True)])
def test_public_interface_resolves_default_and_consumes_results(caller, sse, text):
    caller.sse, caller.text = sse, text
    research = Interface("parallel")
    result = research.search("current research evidence")
    assert isinstance(EngineRepository.get("search"), ParallelMCPEngine)
    assert "Useful current research evidence." in result.value
    assert_citation_contract(result)
    assert result.get_citations()[0].url == "https://example.com/research"
    page = research.scrape(result.get_citations()[0].url, full_content=True)
    assert page.value == "Complete page markdown."
    assert [c["name"] for c in caller.calls] == ["web_search", "web_fetch"]
    assert caller.calls[0]["arguments"]["objective"] == "current research evidence"
    assert len([r for r in caller.requests if r.method == "DELETE"]) == 2


def test_conversations_followups_reset_and_concurrency(caller):
    a, b = Interface("parallel"), Interface("parallel")
    a.search("first related search")
    a.scrape("https://example.com")
    a.search("follow up search")
    initial = [c["arguments"]["session_id"] for c in caller.calls]
    assert len(set(initial)) == 1
    UUID(initial[0])
    engine = EngineRepository.get("search")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda r: r.search("concurrent research query"), [a, b, a, b]))
    ids = [c["arguments"]["session_id"] for c in caller.calls[3:]]
    assert ids.count(a._search_session_id) == ids.count(b._search_session_id) == 2
    assert a._search_session_id != b._search_session_id
    assert EngineRepository.get("search") is engine
    a.reset_search_session()
    a.search("new research run")
    assert caller.calls[-1]["arguments"]["session_id"] not in initial


def test_standalone_decorated_calls_have_independent_context(caller):
    @core.search(query="standalone research")
    def search(_) -> object:
        pass

    search(Symbol("a"))
    search(Symbol("b"))
    assert caller.calls[0]["arguments"]["session_id"] != caller.calls[1]["arguments"]["session_id"]


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, "search"),
        ({"SEARCH_ENGINE_MODEL": "", "SEARCH_ENGINE_API_KEY": ""}, "search"),
        ({"SEARCH_ENGINE_MODEL": "parallel-mcp"}, "search"),
        (
            {"SEARCH_ENGINE_MODEL": "parallel", "SEARCH_ENGINE_API_KEY": "saved"},
            ENGINE_UNREGISTERED,
        ),
        ({"SEARCH_ENGINE_MODEL": "sonar", "SEARCH_ENGINE_API_KEY": "saved"}, ENGINE_UNREGISTERED),
        ({"SEARCH_ENGINE_API_KEY": "saved"}, ENGINE_UNREGISTERED),
        ({"SEARCH_ENGINE_MODEL": "disabled"}, ENGINE_UNREGISTERED),
        ({"SEARCH_ENGINE_MODEL": False}, ENGINE_UNREGISTERED),
        ({"SEARCH_ENGINE_ENABLED": False}, ENGINE_UNREGISTERED),
    ],
)
def test_selection_preserves_choices(monkeypatch, config, expected):
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", config)
    assert ParallelMCPEngine().id() == expected


def test_saved_parallel_provider_uses_existing_route(monkeypatch):
    config = {"SEARCH_ENGINE_MODEL": "parallel", "SEARCH_ENGINE_API_KEY": "saved"}
    for module in ["parallel", "parallel_mcp"]:
        monkeypatch.setattr(f"symai.backend.engines.search.{module}.engine.SYMAI_CONFIG", config)
    assert isinstance(EngineRepository.get("search"), ParallelEngine)


def test_registered_override_is_preserved(caller):
    engine = ParallelEngine(api_key="explicit-key")
    EngineRepository.register("search", engine)
    assert EngineRepository.get("search") is engine
    assert not caller.requests


def test_disabled_search_and_preview_do_not_call_network(monkeypatch, caller):
    monkeypatch.setattr(
        "symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG",
        {"SEARCH_ENGINE_MODEL": "disabled"},
    )
    with pytest.raises(ValueError, match="No engine named search"):
        Interface("parallel").search("disabled query")
    assert not caller.requests
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", {})
    Interface("parallel").search("preview query", preview=True)
    assert not caller.requests


def test_callers_can_supply_explicit_context(caller):
    research = Interface("parallel")
    research.search("explicit context query", session_id="trusted-research-context")
    assert caller.calls[-1]["arguments"]["session_id"] == "trusted-research-context"


@pytest.mark.parametrize(
    "kwargs",
    [{"processor": "base"}, {"mode": "advanced"}, {"allowed_domains": ["gov"]}, {"max_results": 3}],
)
def test_unsupported_api_options_fail_before_network(caller, kwargs):
    with pytest.raises(ValueError, match="does not support"):
        Interface("parallel").search("research query", **kwargs)
    assert not caller.requests


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "https://"])
def test_invalid_url_is_rejected(caller, url):
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        Interface("parallel").scrape(url)
    assert not caller.requests


@pytest.mark.parametrize("kind", ["quota", "timeout", "cancel", "tool", "rpc", "malformed"])
def test_failures_propagate_and_retry_keeps_context(caller, kind):
    research = Interface("parallel")

    def failure(request, body):
        if kind == "quota":
            return httpx.Response(429, headers={"retry-after": "60"}, json={"error": "quota"})
        if kind == "timeout":
            assert request.extensions["timeout"]["read"] == 30
            raise httpx.ReadTimeout("timeout", request=request)
        if kind == "cancel":
            raise asyncio.CancelledError()
        if kind == "rpc":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -1, "message": "failed"},
                },
            )
        if kind == "malformed":
            return httpx.Response(200, json={"id": "other", "result": {}})
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "Tool failure"}]},
            },
        )

    caller.failure = failure
    error = {
        "quota": EngineRateLimitError,
        "timeout": EngineTransportError,
        "cancel": asyncio.CancelledError,
    }.get(kind, ValueError)
    with pytest.raises(error):
        research.search("retry related query")
    assert len(caller.calls) == 1  # No hidden provider fallback or tool replay.
    caller.failure = None
    result = research.search("retry related query")
    assert_citation_contract(result)
    assert caller.calls[0]["arguments"]["session_id"] == caller.calls[1]["arguments"]["session_id"]
    assert len([r for r in caller.requests if r.method == "DELETE"]) == 2


@pytest.mark.parametrize(
    "module,model,cls",
    [
        ("perplexity", "sonar", "PerplexityEngine"),
        ("openai", "gpt-4.1-mini", "OpenAISearchEngine"),
        ("gemini", "gemini-3.5-flash", "GeminiSearchEngine"),
        ("firecrawl", "firecrawl", "FirecrawlEngine"),
    ],
)
def test_saved_other_providers_resolve_normally(monkeypatch, caller, module, model, cls):
    config = {"SEARCH_ENGINE_MODEL": model, "SEARCH_ENGINE_API_KEY": "saved"}
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", config)
    monkeypatch.setattr(f"symai.backend.engines.search.{module}.engine.SYMAI_CONFIG", config)
    assert type(EngineRepository.get("search")).__name__ == cls
    assert not caller.requests


def test_dynamic_override_is_preserved(monkeypatch, caller):
    engine = ParallelEngine(api_key="explicit-key")
    monkeypatch.setattr(EngineRepository(), "get_dynamic_engine_instance", lambda: engine)
    assert EngineRepository.get("search") is engine
    assert not caller.requests


def test_explicit_provider_failure_never_falls_back(caller):
    from symai.backend.transport import EngineAuthenticationError

    engine = ParallelEngine(api_key="explicit-key")
    attempts = []

    def denied(request):
        attempts.append(request)
        return httpx.Response(401, json={"error": "invalid key"})

    with httpx.Client(transport=httpx.MockTransport(denied)) as client:
        engine.transport_client = client
        EngineRepository.register("search", engine)
        with pytest.raises(EngineAuthenticationError):
            Interface("parallel").search("explicit provider query")
    assert len(attempts) == 1
    assert attempts[0].url.host == "api.parallel.ai"
    assert not caller.requests


def test_strict_request_validation_accepts_normal_search(caller):
    result = Interface("parallel").search("strict research query", strict_request_kwargs=True)
    assert_citation_contract(result)


def test_real_dynamic_context_selects_mcp_over_saved_provider(monkeypatch, caller):
    from symai.components import DynamicEngine

    monkeypatch.delattr(EngineRepository(), "get_dynamic_engine_instance")
    config = {"SEARCH_ENGINE_MODEL": "parallel", "SEARCH_ENGINE_API_KEY": "saved"}
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", config)
    monkeypatch.setattr("symai.backend.engines.search.parallel.engine.SYMAI_CONFIG", config)
    original = EngineRepository.get("search")
    assert isinstance(original, ParallelEngine)
    with DynamicEngine("parallel-mcp", "") as selected:
        assert EngineRepository.get("search") is selected
        result = Interface("parallel").search("dynamic research query")
        assert_citation_contract(result)
    assert EngineRepository.get("search") is original
    assert caller.calls[-1]["name"] == "web_search"


def test_explicit_mcp_registration_overrides_saved_provider(monkeypatch, caller):
    monkeypatch.setattr(
        "symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG",
        {"SEARCH_ENGINE_MODEL": "sonar", "SEARCH_ENGINE_API_KEY": "saved"},
    )
    engine = ParallelMCPEngine(client_timeout=12)
    EngineRepository.register("search", engine)
    assert_citation_contract(Interface("parallel").search("registered research query"))
    assert EngineRepository.get("search") is engine
    assert all(r.extensions["timeout"]["read"] == 12 for r in caller.requests)


@pytest.mark.parametrize(
    "config",
    [
        {"SEARCH_ENGINE_ENABLED": False},
        {"SEARCH_ENGINE_MODEL": "disabled"},
        {"SEARCH_ENGINE_MODEL": False},
    ],
)
def test_explicit_registration_keeps_disable_gate(monkeypatch, caller, config):
    monkeypatch.setattr("symai.backend.engines.search.parallel_mcp.engine.SYMAI_CONFIG", config)
    EngineRepository.register("search", ParallelMCPEngine())
    with pytest.raises(ValueError, match="disabled"):
        Interface("parallel").search("disabled registered query")
    assert not caller.requests


def test_explicit_key_is_not_silently_discarded(caller):
    with pytest.raises(ValueError, match="anonymous"):
        ParallelMCPEngine(api_key="explicit-key")
    assert not caller.requests
