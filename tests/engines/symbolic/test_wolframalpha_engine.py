"""Wolfram Alpha engine tests: mock validates the GET wire + pod extraction offline,
live hits the real v2/query API (--engine-api=live, needs WOLFRAM_API_KEY).
"""

import os

import httpx
import pytest

from symai.backend.engines.symbolic.wolframalpha import WolframAlphaEngine, WolframResult
from symai.backend.engines.symbolic.wolframalpha.models import (
    API_PINNED,
    WOLFRAM_QUERY_URL,
)
from symai.core import Argument
from tests.engines.mock_api import MockAPI

CAPITAL_RESPONSE = {
    "queryresult": {
        "success": True,
        "error": False,
        "numpods": 2,
        "pods": [
            {
                "title": "Input interpretation",
                "primary": False,
                "subpods": [{"plaintext": "capital of France", "img": {"src": "x"}}],
            },
            {
                "title": "Result",
                "primary": True,
                "subpods": [{"plaintext": "Paris", "img": {"src": "x"}}],
            },
        ],
    }
}

ERROR_RESPONSE = {
    "queryresult": {
        "success": False,
        "error": {"code": "APPID", "msg": "Invalid appid"},
        "numpods": 0,
        "pods": [],
    }
}


def make_argument(text: str) -> Argument:
    return Argument((), {}, {"processed_input": text})


def run_query(engine: WolframAlphaEngine, text: str):
    argument = make_argument(text)
    engine.prepare(argument)
    results, metadata = engine.forward(argument)
    return results[0], metadata


@pytest.fixture
def engine_api_mode(request):
    return request.config.getoption("--engine-api")


class TestWolframAlphaMock:
    def test_api_pinned_matches_models(self):
        assert API_PINNED == "2026-07-18"

    def test_wire_shape(self):
        engine = WolframAlphaEngine(api_key="dummy-appid")
        with MockAPI(engine, lambda request: httpx.Response(200, json=CAPITAL_RESPONSE)) as mock:
            run_query(engine, "What is the capital of France?")
        request = mock.last_request
        assert request.method == "GET"
        assert str(request.url).startswith(WOLFRAM_QUERY_URL)
        assert request.url.params["input"] == "What is the capital of France?"
        assert request.url.params["appid"] == "dummy-appid"
        assert request.url.params["output"] == "json"
        # GET must not carry a JSON body
        assert not request.content

    def test_parse_extracts_primary_first(self):
        engine = WolframAlphaEngine(api_key="dummy-appid")
        with MockAPI(engine, lambda request: httpx.Response(200, json=CAPITAL_RESPONSE)):
            result, metadata = run_query(engine, "What is the capital of France?")
        assert isinstance(result, WolframResult)
        # the primary "Result" pod leads even though it is second in the payload
        assert str(result).startswith("Paris")
        assert "capital of France" in str(result)
        assert metadata["raw_output"].success is True

    def test_error_response_raises(self):
        engine = WolframAlphaEngine(api_key="dummy-appid")
        with (
            MockAPI(engine, lambda request: httpx.Response(200, json=ERROR_RESPONSE)),
            pytest.raises(ValueError, match="Invalid appid"),
        ):
            run_query(engine, "anything")

    def test_malformed_response_fails_typed_parsing(self):
        engine = WolframAlphaEngine(api_key="dummy-appid")
        with (
            MockAPI(engine, lambda request: httpx.Response(200, json={"queryresult": {}})),
            pytest.raises(Exception, match="success"),
        ):
            run_query(engine, "anything")


class TestWolframAlphaLive:
    @pytest.fixture(autouse=True)
    def require_live(self, engine_api_mode):
        if engine_api_mode != "live":
            pytest.skip("live wolfram test skipped in mock mode")
        if not os.environ.get("WOLFRAM_API_KEY"):
            pytest.skip("WOLFRAM_API_KEY not set; live test skipped")

    def query(self, text: str) -> str:
        engine = WolframAlphaEngine(api_key=os.environ["WOLFRAM_API_KEY"])
        result, _ = run_query(engine, text)
        return str(result)

    def test_live_capital(self):
        assert "Paris" in self.query("What is the capital of France?")

    def test_live_arithmetic(self):
        assert "4" in self.query("What is 2+2?")

    def test_live_integral(self):
        assert "9" in self.query("What is the integral of x^2 from 0 to 3?")
