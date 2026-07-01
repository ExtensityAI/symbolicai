import logging
from typing import Any, ClassVar

import aiohttp

from symai.backend.async_bridge import run_async
from symai.backend.base import Engine
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class LlamaCppEmbeddingEngine(Engine):
    _retry_params: ClassVar[dict[str, Any]] = {
        "tries": 5,
        "delay": 2,
        "max_delay": 60,
        "backoff": 2,
        "jitter": (1, 5),
        "graceful": True,
    }
    _timeout_params: ClassVar[dict[str, Any]] = {
        "read": None,
        "connect": None,
    }

    def __init__(self, retry_params: dict = _retry_params, timeout_params: dict = _timeout_params):
        super().__init__()
        self.config = SYMAI_CONFIG
        if self.id() != "embedding":
            return
        if not SYMSERVER_CONFIG.get("online"):
            msg = "You are using the llama.cpp embedding engine, but the server endpoint is not started. Please start the server with `symserver [--args]`."
            raise ValueError(msg)

        self.server_endpoint = (
            f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"
        )
        self.timeout_params = self._validate_timeout_params(timeout_params)
        self.retry_params = self._validate_retry_params(retry_params)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get("EMBEDDING_ENGINE_MODEL") and self.config.get(
            "EMBEDDING_ENGINE_MODEL"
        ).startswith("llama"):
            return "embedding"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def _validate_timeout_params(self, timeout_params):
        if not isinstance(timeout_params, dict):
            msg = "timeout_params must be a dictionary"
            raise ValueError(msg)
        assert all(key in timeout_params for key in ["read", "connect"]), (
            "Available keys: ['read', 'connect']"
        )
        return timeout_params

    def _validate_retry_params(self, retry_params):
        if not isinstance(retry_params, dict):
            msg = "retry_params must be a dictionary"
            raise ValueError(msg)
        assert all(
            key in retry_params
            for key in ["tries", "delay", "max_delay", "backoff", "jitter", "graceful"]
        ), "Available keys: ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']"
        return retry_params

    async def _arequest(self, text: str, embd_normalize: str) -> dict:
        """Makes an async HTTP request to the llama.cpp server."""

        async def _make_request():
            timeout = aiohttp.ClientTimeout(
                sock_connect=self.timeout_params["connect"], sock_read=self.timeout_params["read"]
            )
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{self.server_endpoint}/v1/embeddings",
                    json={"content": text, "embd_normalize": embd_normalize},
                ) as res,
            ):
                if res.status != 200:
                    msg = f"Request failed with status code: {res.status}"
                    raise ValueError(msg)
                return await res.json()

        return await _make_request()

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        embd_normalize = kwargs.get("embd_normalize", -1)  # -1 = no normalization

        new_dim = kwargs.get("new_dim")
        if new_dim:
            msg = "new_dim is not yet supported"
            raise NotImplementedError(msg)

        try:
            res = run_async(self._arequest(inp, embd_normalize))
        except Exception as e:
            msg = f"Request failed with error: {e!s}"
            raise ValueError(msg) from e

        output = [r["embedding"] for r in res] if res is not None else None  # B x 1 x D
        metadata = {"raw_output": res}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "LlamaCppEmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries
