"""llama.cpp embedding engine tests (local symserver).

The engine builds a fresh per-call httpx.Client and keeps no transport_client
handle, so the shared MockAPI harness cannot inject a MockTransport; the mock
tests below monkeypatch httpx.Client with a MockTransport-backed factory
instead (process-wide but restored by the monkeypatch fixture). That is also
why this engine is not part of tests/engines/embedding/interface.py.

The engine bypasses the shared transport error lattice (plain httpx + manual
status check), so there is no 401 -> EngineAuthenticationError mapping to test:
a non-200 response raises ValueError — pinned by test_non_200_raises_value_error.
"""

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.embedding.llama_cpp.engine import LlamaCppEmbeddingEngine
from symai.backend.engines.embedding.llama_cpp.models import (
    API_PINNED,
    LlamaCppEmbeddingItem,
)
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG
from tests.engines.embedding.interface import normalized_vector

ENTRIES = ["hello", "world"]
MOCK_VECTORS = [normalized_vector(0.11), normalized_vector(0.07)]
# Captured at import, before the engine fixture monkeypatches the config dicts —
# the live gate must reflect the real symserver state, not the fixture's.
SERVER_ONLINE = bool(SYMSERVER_CONFIG.get("online"))


@pytest.fixture
def engine(monkeypatch):
    # id() == "embedding" requires a llama* configured model; construction then
    # requires the symserver endpoint to be flagged online.
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_MODEL", "llama-embedding")
    monkeypatch.setitem(SYMSERVER_CONFIG, "online", True)
    monkeypatch.setitem(SYMSERVER_CONFIG, "--host", "localhost")
    monkeypatch.setitem(SYMSERVER_CONFIG, "--port", 8000)
    return LlamaCppEmbeddingEngine()


@pytest.fixture
def mock_wire(monkeypatch):
    """Route the engine's per-call httpx.Client through a MockTransport; record requests."""
    requests = []
    real_client = httpx.Client

    def factory(*_args, **kwargs):
        def spy(request):
            requests.append(request)
            return httpx.Response(
                200, json=[{"embedding": v} for v in MOCK_VECTORS], request=request
            )

        return real_client(transport=httpx.MockTransport(spy), timeout=kwargs.get("timeout"))

    monkeypatch.setattr(httpx, "Client", factory)
    return requests


def make_argument(entries=None, kwargs=None):
    return SimpleNamespace(
        prop=SimpleNamespace(
            entries=list(entries if entries is not None else ENTRIES),
            prepared_input=None,
            processed_input=None,
        ),
        kwargs=kwargs or {},
    )


def test_api_pinned_declared():
    assert API_PINNED, "llama_cpp embedding models.py must declare API_PINNED"


def test_prepare_produces_expected_prepared_input(engine):
    argument = make_argument()

    engine.prepare(argument)

    assert argument.prop.prepared_input == ENTRIES


def test_forward_mock_returns_vectors(engine, mock_wire):
    argument = make_argument()
    engine.prepare(argument)

    output, metadata = engine.forward(argument)

    request = mock_wire[-1]
    assert request.method == "POST"
    assert str(request.url) == f"{engine.server_endpoint}/v1/embeddings"
    assert json.loads(request.content.decode("utf-8")) == {
        "content": ENTRIES,
        "embd_normalize": -1,
    }
    vectors = output[0]
    assert len(vectors) == len(ENTRIES)
    assert all(len(v) == len(MOCK_VECTORS[0]) for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)
    assert all(isinstance(item, LlamaCppEmbeddingItem) for item in metadata["raw_output"])


def test_new_dim_raises_not_implemented(engine):
    argument = make_argument(kwargs={"new_dim": 128})
    engine.prepare(argument)

    with pytest.raises(NotImplementedError, match="new_dim"):
        engine.forward(argument)


@pytest.fixture
def fast_retry_engine(engine):
    # Same patched config as `engine`, but with zero-delay retries so failure-path
    # tests stay fast. Unknown keys are rejected; partial overrides merge over defaults.
    engine.retry_params = {"tries": 5, "delay": 0, "max_delay": 0, "backoff": 1, "jitter": (0, 0)}
    return engine


def mock_wire_factory(monkeypatch, handler):
    """Route the engine's per-call httpx.Client through handler; record requests."""
    requests = []
    real_client = httpx.Client

    def factory(*_args, **kwargs):
        def spy(request):
            requests.append(request)
            return handler(request)

        return real_client(transport=httpx.MockTransport(spy), timeout=kwargs.get("timeout"))

    monkeypatch.setattr(httpx, "Client", factory)
    return requests


def test_retries_transport_errors_then_succeeds(fast_retry_engine, monkeypatch):
    # Regression: the folder-ized engine called httpx bare (single attempt); the old
    # engine wrapped calls in tries=5/delay=2/backoff=2 retries. A flaky local server
    # (connection refused while reloading) must be retried.
    state = {"failures": 0}

    def handler(request):
        if state["failures"] < 2:
            state["failures"] += 1
            msg = "connection refused"
            raise httpx.ConnectError(msg, request=request)
        return httpx.Response(200, json=[{"embedding": v} for v in MOCK_VECTORS], request=request)

    requests = mock_wire_factory(monkeypatch, handler)
    argument = make_argument()
    fast_retry_engine.prepare(argument)

    output, _metadata = fast_retry_engine.forward(argument)

    assert len(requests) == 3  # 2 failures + 1 success
    assert len(output[0]) == len(ENTRIES)


def test_retries_5xx_then_succeeds(fast_retry_engine, monkeypatch):
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(500, json={"error": "model loading"}, request=request)
        return httpx.Response(200, json=[{"embedding": v} for v in MOCK_VECTORS], request=request)

    requests = mock_wire_factory(monkeypatch, handler)
    argument = make_argument()
    fast_retry_engine.prepare(argument)

    output, _metadata = fast_retry_engine.forward(argument)

    assert len(requests) == 2
    assert len(output[0]) == len(ENTRIES)


def test_retry_exhaustion_raises_value_error(fast_retry_engine, monkeypatch):
    def handler(request):
        msg = "connection refused"
        raise httpx.ConnectError(msg, request=request)

    requests = mock_wire_factory(monkeypatch, handler)
    argument = make_argument()
    fast_retry_engine.prepare(argument)

    with pytest.raises(ValueError, match="Request failed with error"):
        fast_retry_engine.forward(argument)

    assert len(requests) == fast_retry_engine.retry_params["tries"]


def test_4xx_is_not_retried(fast_retry_engine, monkeypatch):
    # Deliberate divergence from the old engine (which retried everything): a local
    # server's 4xx is deterministic and retrying cannot heal it.
    requests = mock_wire_factory(
        monkeypatch,
        lambda request: httpx.Response(401, json={"error": "unauthorized"}, request=request),
    )
    argument = make_argument()
    fast_retry_engine.prepare(argument)

    with pytest.raises(ValueError, match="status code: 401"):
        fast_retry_engine.forward(argument)

    assert len(requests) == 1


def test_retry_params_validation(engine):
    with pytest.raises(ValueError, match="Unknown retry_params keys"):
        type(engine)(retry_params={"bogus": 1})
    with pytest.raises(ValueError, match="must be a dictionary"):
        type(engine)(retry_params=["tries"])


def test_malformed_response_fails_typed_parsing(engine, monkeypatch):
    real_client = httpx.Client

    def factory(*_args, **kwargs):
        # error payload is an object, not the bare array of embedding items
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"error": "boom"}, request=request)
            ),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(httpx, "Client", factory)
    argument = make_argument()
    engine.prepare(argument)

    with pytest.raises(ValidationError):
        engine.forward(argument)


def test_non_200_raises_value_error(engine, monkeypatch):
    # NOTE: no shared transport error lattice here — non-200 maps to ValueError.
    real_client = httpx.Client

    def factory(*_args, **kwargs):
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={"error": "unauthorized"}, request=request)
            ),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(httpx, "Client", factory)
    argument = make_argument()
    engine.prepare(argument)

    with pytest.raises(ValueError, match="status code: 401"):
        engine.forward(argument)


@pytest.mark.engine_live
def test_live_smoke(engine, engine_api_mode):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live embedding requests")
    if not SERVER_ONLINE:
        pytest.skip("llama.cpp server is not online (start it with symserver)")

    argument = make_argument(entries=["hello"])
    engine.prepare(argument)
    try:
        output, metadata = engine.forward(argument)
    except ValueError as e:
        # The running symserver build may not expose the embedding endpoint
        # (501/404) — that is an environment condition, not an engine failure.
        if "status code: 501" in str(e) or "status code: 404" in str(e):
            pytest.skip(f"symserver does not serve /v1/embeddings: {e}")
        raise

    vectors = output[0]
    assert len(vectors) == 1
    assert isinstance(vectors[0], list) and len(vectors[0]) > 0
    assert all(isinstance(x, float) for x in vectors[0])
    assert all(isinstance(item, LlamaCppEmbeddingItem) for item in metadata["raw_output"])
