"""BFL FLUX drawing engine tests: mock wire replay + live smoke (--engine-api=live)."""

from __future__ import annotations

import httpx
import pytest

from symai.backend.engines.drawing.bfl.engine import DrawingEngine
from symai.backend.engines.drawing.bfl.models import API_PINNED, BFL_API_BASE, BflPollResponse
from tests.engines.drawing.interface import MOCK_PNG_BYTES, MOCK_PROMPT, DrawingEngineTestInterface
from tests.engines.mock_api import MockAPI

MOCK_TASK_ID = "mock-task-id"
MOCK_POLL_URL = f"{BFL_API_BASE}/get_result?id={MOCK_TASK_ID}"
MOCK_SAMPLE_URL = "https://cdn.bfl.ai/mock/sample.png"


class TestBflDrawingEngine(DrawingEngineTestInterface):
    engine_cls = DrawingEngine
    default_model = "flux-pro-1.1"
    wire_url = f"{BFL_API_BASE}/flux-pro-1.1"
    auth_header_name = "x-key"
    auth_header_prefix = ""
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.drawing.bfl.models"
    api_key_env = "BFL_API_KEY"

    def mock_handler(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"id": MOCK_TASK_ID, "polling_url": MOCK_POLL_URL},
                    request=request,
                )
            if request.url.path.endswith("/get_result"):
                return httpx.Response(
                    200,
                    json={
                        "id": MOCK_TASK_ID,
                        "status": "Ready",
                        "result": {"sample": MOCK_SAMPLE_URL},
                    },
                    request=request,
                )
            return httpx.Response(200, content=MOCK_PNG_BYTES, request=request)

        return handler

    def expected_request_body_subset(self) -> dict:
        return {
            "prompt": MOCK_PROMPT,
            "width": 1024,
            "height": 768,
            "num_inference_steps": 40,
            "safety_tolerance": 2,
        }

    def mock_forward_kwargs(self) -> dict:
        return {"operation": "create"}

    def live_forward_kwargs(self) -> dict:
        return {"width": 512, "height": 512}

    def configure_mock_engine(self, engine) -> None:
        engine.poll_interval_seconds = 0

    def assert_raw_output(self, metadata: dict):
        assert isinstance(metadata["raw_output"], BflPollResponse)

    def test_forward_mock_polls_then_downloads(self):
        """The wire sequence is submit POST -> poll GET -> image GET."""
        _api, output, _metadata = self.forward_through_mock()

        methods = [(request.method, request.url.path) for request in _api.requests]
        assert methods[0] == ("POST", f"/v1/{self.default_model}")
        assert any(path.endswith("/get_result") for _method, path in methods[1:])
        assert output[0].value

    def test_poll_pending_transitions_to_ready(self):
        """A Pending poll response keeps the loop polling until Ready."""
        polls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"id": MOCK_TASK_ID, "polling_url": MOCK_POLL_URL},
                    request=request,
                )
            if request.url.path.endswith("/get_result"):
                polls["count"] += 1
                status = "Pending" if polls["count"] < 3 else "Ready"
                result = {"sample": MOCK_SAMPLE_URL} if status == "Ready" else None
                return httpx.Response(
                    200,
                    json={"id": MOCK_TASK_ID, "status": status, "result": result},
                    request=request,
                )
            return httpx.Response(200, content=MOCK_PNG_BYTES, request=request)

        _api, output, _metadata = self.forward_through_mock(handler=handler)

        assert polls["count"] == 3
        assert output[0].value

    @pytest.mark.parametrize(
        "status", ["Error", "Task not found", "Request Moderated", "Content Moderated"]
    )
    def test_poll_failure_status_raises(self, status):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"id": MOCK_TASK_ID, "polling_url": MOCK_POLL_URL},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"id": MOCK_TASK_ID, "status": status, "result": None},
                request=request,
            )

        engine = self.make_engine()
        self.configure_mock_engine(engine)
        with MockAPI(engine, handler):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises(RuntimeError, match=status):
                engine.forward(argument)

    def test_unknown_operation_raises(self):
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"operation": "edit"})

        with pytest.raises(Exception, match="Unknown operation"):
            engine.forward(argument)
