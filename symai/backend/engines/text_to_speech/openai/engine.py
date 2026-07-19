from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from symai.backend.base import Engine
from symai.backend.engines.text_to_speech.openai.models import (
    OPENAI_SPEECH_URL,
    OpenAISpeechRequest,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

if TYPE_CHECKING:
    import httpx

silence_noisy_loggers("openai")

logger = logging.getLogger(__name__)


class TTSEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config["TEXT_TO_SPEECH_ENGINE_API_KEY"] if api_key is None else api_key
        self.model_id = self.config["TEXT_TO_SPEECH_ENGINE_MODEL"] if model is None else model
        self.tokens = []
        self.text = []
        self.name = self.__class__.__name__
        self.transport_client = None

        if api_key is None and model is None and self.id() != "text-to-speech":
            return  # do not initialize if not text-to-speech; see EngineRepository.register_from_package

    def id(self) -> str:
        if self.config["TEXT_TO_SPEECH_ENGINE_API_KEY"]:
            return "text-to-speech"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "TEXT_TO_SPEECH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["TEXT_TO_SPEECH_ENGINE_API_KEY"]
        if "TEXT_TO_SPEECH_ENGINE_MODEL" in kwargs:
            self.model_id = kwargs["TEXT_TO_SPEECH_ENGINE_MODEL"]

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response, argument)

    def build_request(self, argument) -> EngineAPIRequest:
        voice, _path, prompt = argument.prop.prepared_input
        kwargs = argument.kwargs
        payload = OpenAISpeechRequest(
            model=self.model_id,
            input=prompt,
            voice=voice,
            response_format=kwargs.get("response_format"),
            speed=kwargs.get("speed"),
        )
        return EngineAPIRequest(
            provider="openai",
            operation="speech",
            payload=payload,
            method="POST",
            url=OPENAI_SPEECH_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> httpx.Response:
        # NOTE: success is audio BYTES (content type follows response_format); the shared
        # transport raises the typed error lattice for non-2xx, whose body IS json.
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        return execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )

    def parse_response(self, response: httpx.Response, argument):
        _voice, path, _prompt = argument.prop.prepared_input
        audio = response.content
        Path(path).write_bytes(audio)
        metadata = {
            "raw_output": response,
            "content_type": response.headers.get("content-type"),
        }
        return [Result(audio)], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "TTSEngine does not support processed_input."
        assert "voice" in argument.kwargs, "TTS requires voice selection."
        assert "path" in argument.kwargs, "TTS requires path selection."
        voice = str(argument.kwargs["voice"]).lower()
        audio_file = str(argument.kwargs["path"])
        prompt = str(argument.prop.prompt)
        argument.prop.prepared_input = (voice, audio_file, prompt)
