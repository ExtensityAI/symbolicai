import logging

from openai import OpenAI
from typing import Optional

# suppress openai logging
logging.getLogger("openai").setLevel(logging.WARNING)

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


class TTSEngine(Engine):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.config   = SYMAI_CONFIG
        self.api_key  = self.config['TEXT_TO_SPEECH_ENGINE_API_KEY'] if api_key is None else api_key
        self.model_id = self.config['TEXT_TO_SPEECH_ENGINE_MODEL'] if model is None else model
        self.tokens   = []
        self.text     = []
        self.client   = OpenAI(api_key=self.api_key)

    def id(self) -> str:
        if self.config['TEXT_TO_SPEECH_ENGINE_API_KEY']:
            return 'text-to-speech'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'TEXT_TO_SPEECH_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['TEXT_TO_SPEECH_ENGINE_API_KEY']
        if 'TEXT_TO_SPEECH_ENGINE_MODEL' in kwargs:
            self.model_id = kwargs['TEXT_TO_SPEECH_ENGINE_MODEL']

    def forward(self, argument):
        kwargs              = argument.kwargs
        voice, path, prompt = argument.prop.prepared_input

        rsp = self.client.audio.speech.create(
            model=self.model_id,
            voice=voice,
            input=prompt
        )

        metadata = {}

        rsp.stream_to_file(path)

        rsp = Result(rsp)
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "TTSEngine does not support processed_input."
        assert 'voice' in argument.kwargs, "TTS requires voice selection."
        assert 'path' in argument.kwargs, "TTS requires path selection."
        voice       = str(argument.kwargs['voice']).lower()
        audio_file  = str(argument.kwargs['path'])
        prompt      = str(argument.prop.prompt)
        argument.prop.prepared_input = (voice, audio_file, prompt)
