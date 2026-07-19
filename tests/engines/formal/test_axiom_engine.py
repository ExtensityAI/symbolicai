"""Axiom engine tests: mock validates the NDJSON wire offline, live runs real
proof tools (--engine-api=live, needs AXIOM_API_KEY).
"""

import os

import httpx
import pytest

from symai.backend.engines.formal.axiom import AxiomEngine, AxiomResult
from symai.backend.engines.formal.axiom.models import (
    API_PINNED,
    AXLE_API_BASE,
)
from symai.backend.transport import EngineAuthenticationError
from symai.core import Argument
from tests.engines.mock_api import MockAPI

OK_LINE = '{"okay": true, "content": "checked", "lean_messages": {}, "tool_messages": {}, "failed_declarations": [], "timings": {"total": 42}, "info": null}\n'
USER_ERROR_LINE = '{"user_error": "invalid Lean syntax"}\n'
INTERNAL_ERROR_LINE = '{"internal_error": "worker exploded"}\n'

LEAN4_VALID = """
theorem hello_world (a b : Prop) (ha : a) (hb : b) : a ∧ b := by
  exact ⟨ha, hb⟩
""".strip()

LEAN4_THEOREM = """
theorem my_zero_add (n : Nat) : 0 + n = n := by
  omega
""".strip()

LEAN4_FALSE_THEOREM = """
theorem false_claim (n : Nat) : n + 1 = n := by
  omega
""".strip()


def make_argument(text: str, **kwargs) -> Argument:
    return Argument((), {}, {"processed_input": text, **kwargs})


def run_tool(engine: AxiomEngine, text: str, **kwargs):
    argument = make_argument(text, **kwargs)
    engine.prepare(argument)
    results, metadata = engine.forward(argument)
    return results[0], metadata


@pytest.fixture
def engine_api_mode(request):
    return request.config.getoption("--engine-api")


class TestAxiomMock:
    def test_api_pinned_matches_models(self):
        assert API_PINNED == "2026-07-18"

    def test_wire_shape(self):
        engine = AxiomEngine(api_key="dummy-key")
        with MockAPI(engine, lambda _request: httpx.Response(200, text=OK_LINE)) as mock:
            run_tool(engine, LEAN4_VALID, tool="check")
        request = mock.last_request
        assert request.method == "POST"
        assert str(request.url) == f"{AXLE_API_BASE}/api/v1/check"
        assert request.headers["Authorization"] == "Bearer dummy-key"
        body = mock.last_body
        assert body["content"] == LEAN4_VALID
        # defaults applied: latest Lean 4 environment, standalone-snippet import leniency
        assert body["environment"] == "lean-4.28.0"
        assert body["ignore_imports"] is True

    def test_parse_ok(self):
        engine = AxiomEngine(api_key="dummy-key")
        with MockAPI(engine, lambda _request: httpx.Response(200, text=OK_LINE)):
            result, metadata = run_tool(engine, LEAN4_VALID, tool="check")
        assert isinstance(result, AxiomResult)
        assert result.raw["okay"] is True
        assert result.raw["timings"] == {"total": 42}
        assert metadata["raw_output"].okay is True

    def test_verify_proof_extra_body(self):
        engine = AxiomEngine(api_key="dummy-key")
        with MockAPI(engine, lambda _request: httpx.Response(200, text=OK_LINE)) as mock:
            run_tool(
                engine,
                LEAN4_THEOREM,
                tool="verify_proof",
                config={
                    "formal_statement": "theorem my_zero_add (n : Nat) : 0 + n = n := by sorry"
                },
            )
        body = mock.last_body
        assert body["formal_statement"].startswith("theorem my_zero_add")
        assert str(mock.last_request.url).endswith("/verify_proof")

    def test_merge_uses_documents(self):
        engine = AxiomEngine(api_key="dummy-key")
        with MockAPI(engine, lambda _request: httpx.Response(200, text=OK_LINE)) as mock:
            run_tool(engine, LEAN4_VALID, tool="merge")
        body = mock.last_body
        assert body["documents"] == [LEAN4_VALID]
        # merge replaces content with documents; None fields never reach the wire
        assert "content" not in body

    def test_unknown_tool_raises(self):
        engine = AxiomEngine(api_key="dummy-key")
        argument = make_argument(LEAN4_VALID, tool="not_a_tool")
        engine.prepare(argument)
        with pytest.raises(ValueError, match="Unknown tool"):
            engine.forward(argument)

    def test_user_error_raises_value_error(self):
        engine = AxiomEngine(api_key="dummy-key")
        with (
            MockAPI(engine, lambda _request: httpx.Response(200, text=USER_ERROR_LINE)),
            pytest.raises(ValueError, match="invalid Lean syntax"),
        ):
            run_tool(engine, LEAN4_VALID, tool="check")

    def test_internal_error_raises_runtime_error(self):
        engine = AxiomEngine(api_key="dummy-key")
        with (
            MockAPI(engine, lambda _request: httpx.Response(200, text=INTERNAL_ERROR_LINE)),
            pytest.raises(RuntimeError, match="worker exploded"),
        ):
            run_tool(engine, LEAN4_VALID, tool="check")

    def test_multi_line_ndjson_raises(self):
        engine = AxiomEngine(api_key="dummy-key")
        with (
            MockAPI(engine, lambda _request: httpx.Response(200, text=OK_LINE + OK_LINE)),
            pytest.raises(RuntimeError, match="Expected 1 response line"),
        ):
            run_tool(engine, LEAN4_VALID, tool="check")

    def test_http_401_raises_authentication_error(self):
        engine = AxiomEngine(api_key="bad-key")
        with (
            MockAPI(
                engine,
                lambda _request: httpx.Response(
                    401, json={"error": {"code": "invalid_api_key", "message": "nope"}}
                ),
            ),
            pytest.raises(EngineAuthenticationError),
        ):
            run_tool(engine, LEAN4_VALID, tool="check")


class TestAxiomLive:
    @pytest.fixture(autouse=True)
    def require_live(self, engine_api_mode):
        if engine_api_mode != "live":
            pytest.skip("live axiom test skipped in mock mode")
        if not os.environ.get("AXIOM_API_KEY"):
            pytest.skip("AXIOM_API_KEY not set; live test skipped")

    def query(self, text: str, **kwargs) -> AxiomResult:
        engine = AxiomEngine(api_key=os.environ["AXIOM_API_KEY"])
        result, _ = run_tool(engine, text, **kwargs)
        return result

    def test_live_check(self):
        result = self.query(LEAN4_VALID, tool="check")
        assert result.raw["okay"] is True

    def test_live_verify_proof(self):
        result = self.query(
            LEAN4_THEOREM,
            tool="verify_proof",
            config={"formal_statement": "theorem my_zero_add (n : Nat) : 0 + n = n := by sorry"},
        )
        assert result.raw["okay"] is True

    def test_live_disprove(self):
        result = self.query(LEAN4_FALSE_THEOREM, tool="disprove")
        # disprove succeeds when it finds a counterexample OR refutes the claim;
        # the wire contract is a well-formed response, not a specific verdict
        assert "okay" in result.raw

    def test_live_extract_theorems(self):
        result = self.query(LEAN4_THEOREM, tool="extract_theorems")
        assert "okay" in result.raw
