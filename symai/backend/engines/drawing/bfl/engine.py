from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from symai.backend.base import Engine
from symai.backend.engines.drawing.bfl.models import (
    BFL_API_BASE,
    BFL_FAILURE_STATUSES,
    BFL_GET_RESULT_URL,
    BFL_POLL_INTERVAL_SECONDS,
    BflGetResultRequest,
    BflImageRequest,
    BflPollResponse,
    BflSubmitResponse,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import (
    DEFAULT_RETRIES,
    default_engine_api_client,
    execute_engine_api_request,
)
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class FluxResult(Result):
    def __init__(self, value: BflPollResponse | dict, image_path: str, **kwargs):
        raw_dict = value.model_dump() if hasattr(value, "model_dump") else value
        super().__init__(raw_dict, **kwargs)
        # unpack the result: the image bytes are already downloaded to a local file
        self._value = [image_path]


class DrawingEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config["DRAWING_ENGINE_API_KEY"] if api_key is None else api_key
        self.model = self.config["DRAWING_ENGINE_MODEL"] if model is None else model
        self.name = self.__class__.__name__
        self.transport_client = None
        self.poll_interval_seconds = BFL_POLL_INTERVAL_SECONDS

    def id(self) -> str:
        if self.config["DRAWING_ENGINE_API_KEY"] and self.config["DRAWING_ENGINE_MODEL"].startswith(
            "flux"
        ):
            return "drawing"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "DRAWING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["DRAWING_ENGINE_API_KEY"]
        if "DRAWING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["DRAWING_ENGINE_MODEL"]

    def forward(self, argument):
        kwargs = argument.kwargs
        except_remedy = kwargs.get("except_remedy", None)

        if kwargs.get("operation") == "create":
            try:
                request = self.build_request(argument)
                response = self.call_request(request)
                rsp, metadata = self.parse_response(response)
            except Exception as e:
                if except_remedy is None:
                    raise e
                rsp = except_remedy(self, e, None, argument)
                metadata = {}
            return [rsp], metadata
        msg = f"Unknown operation: {kwargs['operation']}"
        raise Exception(msg)

    def build_request(self, argument) -> EngineAPIRequest:
        kwargs = argument.kwargs
        payload = BflImageRequest.model_validate(
            {
                "prompt": argument.prop.prepared_input,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 768),
                "num_inference_steps": kwargs.get("steps", 40),
                "guidance_scale": kwargs.get("guidance", None),
                "seed": kwargs.get("seed", None),
                "safety_tolerance": kwargs.get("safety_tolerance", 2),
            }
        )
        return EngineAPIRequest(
            provider="bfl",
            operation="create",
            payload=payload,
            method="POST",
            url=f"{BFL_API_BASE}/{self.model}",
            headers={
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> BflPollResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        submit_response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        submit = BflSubmitResponse.model_validate(submit_response.json())
        if not submit.id:
            msg = f"Failed to get request ID! Response payload: {submit_response.json()}"
            raise ValueError(msg)

        # NOTE: BFL requires polling the polling_url returned by the submit response
        # (the task id is embedded in it); fall back to get_result?id=... otherwise.
        poll_url = submit.polling_url or BFL_GET_RESULT_URL
        poll_params = None if submit.polling_url else {"id": submit.id}
        while True:
            time.sleep(self.poll_interval_seconds)
            poll_request = EngineAPIRequest(
                provider="bfl",
                operation="get_result",
                payload=BflGetResultRequest(),
                method="GET",
                url=poll_url,
                headers=request.headers,
                params=poll_params,
                timeout=self.client_timeout,
            )
            poll_response = execute_engine_api_request(
                poll_request,
                client=self.transport_client,
                max_retries=max_retries,
            )
            poll = BflPollResponse.model_validate(poll_response.json())

            if poll.status == "Ready":
                return poll
            if poll.status in BFL_FAILURE_STATUSES:
                msg = f"Flux generation failed with status '{poll.status}': {poll_response.json()}"
                raise RuntimeError(msg)

    def parse_response(self, response: BflPollResponse):
        sample_url = response.result.sample if response.result else None
        if not sample_url:
            msg = f"Flux generation returned status 'Ready' without a sample URL: {response.model_dump()}"
            raise ValueError(msg)
        image_path = self._download_image(sample_url)
        return FluxResult(response, image_path), {"raw_output": response}

    def _download_image(self, url: str) -> str:
        client = self.transport_client or default_engine_api_client()
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            path = tmp_file.name
        with Path(path).open("wb") as f:
            f.write(response.content)
        return path

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
