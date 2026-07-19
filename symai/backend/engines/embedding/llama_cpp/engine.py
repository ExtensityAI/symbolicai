import logging
import random
import time
from typing import ClassVar

import httpx
from pydantic import TypeAdapter

from symai.backend.base import Engine
from symai.backend.engines.embedding.llama_cpp.models import LlamaCppEmbeddingResponse
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class LlamaCppEmbeddingEngine(Engine):
    # NOTE: retry policy restored from the old aiohttp engine (core_ext.retry with
    # tries=5, delay=2, max_delay=60, backoff=2, jitter=(1,5)). One divergence: the old
    # engine retried *every* failure including deterministic 4xx; this one retries
    # transport errors and 5xx only (a local server's 4xx never heals by waiting).
    _retry_params: ClassVar[dict] = {
        "tries": 5,
        "delay": 2,
        "max_delay": 60,
        "backoff": 2,
        "jitter": (1, 5),
    }
    _timeout_params: ClassVar[dict] = {
        "read": None,
        "connect": None,
    }

    def __init__(self, retry_params: dict | None = None, timeout_params: dict = _timeout_params):
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
        self.retry_params = self._validate_retry_params(retry_params)
        self.timeout_params = self._validate_timeout_params(timeout_params)
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

    def _validate_retry_params(self, retry_params):
        if retry_params is None:
            return dict(self._retry_params)
        if not isinstance(retry_params, dict):
            msg = "retry_params must be a dictionary"
            raise ValueError(msg)
        # Caller overrides merge over the defaults; unknown keys are rejected.
        unknown = set(retry_params) - set(self._retry_params)
        if unknown:
            msg = f"Unknown retry_params keys: {sorted(unknown)}. Available keys: {sorted(self._retry_params)}"
            raise ValueError(msg)
        return {**self._retry_params, **retry_params}

    def _validate_timeout_params(self, timeout_params):
        if not isinstance(timeout_params, dict):
            msg = "timeout_params must be a dictionary"
            raise ValueError(msg)
        assert all(key in timeout_params for key in ["read", "connect"]), (
            "Available keys: ['read', 'connect']"
        )
        return timeout_params

    def _post_embeddings(self, inp: list, embd_normalize: int) -> list:
        """POST to the local server with exponential-backoff retries.

        Retries transport errors and 5xx (the local server is flaky while loading
        models); raises ValueError after the last attempt or immediately on 4xx.
        """
        params = self.retry_params
        tries = max(1, int(params["tries"]))
        delay = params["delay"]
        jitter = params["jitter"]
        timeout = httpx.Timeout(
            timeout=None,
            connect=self.timeout_params["connect"],
            read=self.timeout_params["read"],
            write=None,
            pool=None,
        )
        last_error: ValueError | None = None
        for attempt in range(tries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{self.server_endpoint}/v1/embeddings",
                        json={"content": inp, "embd_normalize": embd_normalize},
                    )
            except httpx.HTTPError as e:
                last_error = ValueError(f"Request failed with error: {e!s}")
            else:
                if response.status_code == 200:
                    return TypeAdapter(LlamaCppEmbeddingResponse).validate_python(response.json())
                last_error = ValueError(f"Request failed with status code: {response.status_code}")
                if response.status_code < 500:
                    raise last_error  # deterministic client error — retrying cannot heal it
            if attempt + 1 < tries:
                time.sleep(delay + random.uniform(*jitter))
                delay = min(delay * params["backoff"], params["max_delay"])
        raise last_error

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        embd_normalize = kwargs.get("embd_normalize", -1)  # -1 = no normalization

        new_dim = kwargs.get("new_dim")
        if new_dim:
            msg = "new_dim is not yet supported"
            raise NotImplementedError(msg)

        res = self._post_embeddings(inp, embd_normalize)

        output = [item.embedding for item in res]  # B x 1 x D
        metadata = {"raw_output": res}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "LlamaCppEmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries
