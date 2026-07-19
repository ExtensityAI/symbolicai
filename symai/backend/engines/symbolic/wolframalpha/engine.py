import logging
from copy import deepcopy

from symai.backend.base import Engine
from symai.backend.engines.symbolic.wolframalpha.models import (
    WOLFRAM_QUERY_URL,
    WolframQueryParams,
    WolframQueryResult,
    WolframResponse,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import execute_engine_api_request
from symai.symbol import Result

logger = logging.getLogger(__name__)


class WolframResult(Result):
    def __init__(self, value, raw: WolframQueryResult) -> None:
        super().__init__(value)
        self.raw = raw
        self._value = value


class WolframAlphaEngine(Engine):
    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = self.config["SYMBOLIC_ENGINE_API_KEY"] if api_key is None else api_key
        self.name = self.__class__.__name__
        self.transport_client = None

    def id(self) -> str:
        if self.config["SYMBOLIC_ENGINE_API_KEY"]:
            return "symbolic"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SYMBOLIC_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SYMBOLIC_ENGINE_API_KEY"]

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def build_request(self, argument) -> EngineAPIRequest:
        # NOTE: the v2 API is GET-only; the payload rides as query params, not a body
        params = WolframQueryParams(input=argument.prop.prepared_input, appid=self.api_key)
        return EngineAPIRequest(
            provider="wolframalpha",
            operation="query",
            payload=params,
            method="GET",
            url=WOLFRAM_QUERY_URL,
            params=params.model_dump(exclude_none=True),
        )

    def call_request(self, request: EngineAPIRequest) -> WolframResponse:
        response = execute_engine_api_request(request, client=self.transport_client)
        return WolframResponse.model_validate(response.json())

    def parse_response(self, response: WolframResponse) -> tuple[list[Result], dict]:
        result = response.queryresult
        if not result.success or result.error:
            msg = f"Failed to interact with WolframAlpha: {result.error!r} (didyoumeans={result.didyoumeans!r})"
            raise ValueError(msg)
        # primary pods first: they carry the direct answer (e.g. the "Result" pod)
        ordered = sorted(result.pods, key=lambda pod: not pod.primary)
        texts = [subpod.plaintext for pod in ordered for subpod in pod.subpods if subpod.plaintext]
        return [WolframResult("\n".join(texts), raw=result)], {"raw_output": result}

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)
