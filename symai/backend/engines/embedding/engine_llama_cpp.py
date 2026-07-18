import logging
from typing import ClassVar

import httpx

from symai.backend.base import Engine
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class LlamaCppEmbeddingEngine(Engine):
    _timeout_params: ClassVar[dict] = {
        "read": None,
        "connect": None,
    }

    def __init__(self, timeout_params: dict = _timeout_params):
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

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        embd_normalize = kwargs.get("embd_normalize", -1)  # -1 = no normalization

        new_dim = kwargs.get("new_dim")
        if new_dim:
            msg = "new_dim is not yet supported"
            raise NotImplementedError(msg)

        timeout = httpx.Timeout(
            timeout=None,
            connect=self.timeout_params["connect"],
            read=self.timeout_params["read"],
            write=None,
            pool=None,
        )
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.server_endpoint}/v1/embeddings",
                    json={"content": inp, "embd_normalize": embd_normalize},
                )
        except httpx.HTTPError as e:
            msg = f"Request failed with error: {e!s}"
            raise ValueError(msg) from e
        if response.status_code != 200:
            msg = f"Request failed with status code: {response.status_code}"
            raise ValueError(msg)
        res = response.json()

        output = [r["embedding"] for r in res] if res is not None else None  # B x 1 x D
        metadata = {"raw_output": res}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "LlamaCppEmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries
