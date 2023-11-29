import logging
from openai import OpenAI

# suppress openai logging
logging.getLogger("openai").setLevel(logging.WARNING)

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


class TTSEngine(Engine):
    def __init__(self):
        super().__init__()
        self.config   = SYMAI_CONFIG
        self.api_key  = self.config['TEXT_TO_SPEECH_ENGINE_API_KEY']
        self.model_id = self.config['TEXT_TO_SPEECH_ENGINE_MODEL']
        self.tokens   = []
        self.text     = []
        self.client   = OpenAI(api_key=self.api_key)

    def id(self) -> str:
        if self.config['TEXT_TO_SPEECH_ENGINE_API_KEY']:
            return 'text-to-speech'
        return super().id() # default to unregistered

    def command(self, argument):
        super().command(argument.kwargs)
        if 'TEXT_TO_SPEECH_ENGINE_API_KEY' in argument.kwargs:
            self.api_key = argument.kwargs['TEXT_TO_SPEECH_ENGINE_API_KEY']
        if 'TEXT_TO_SPEECH_ENGINE_MODEL' in argument.kwargs:
            self.model_id = argument.kwargs['TEXT_TO_SPEECH_ENGINE_MODEL']

    def forward(self, argument):
        kwargs              = argument.kwargs
        voice, path, prompt = argument.prop.processed_input

        input_handler       = kwargs.get("input_handler")
        if input_handler is not None:
            input_handler((prompt, voice, path))

        rsp = self.client.audio.speech.create(
            model=self.model_id,
            voice=voice,
            input=prompt
        )

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (prompt, voice, path)
            metadata['output'] = rsp
            metadata['model']  = self.model_id
            metadata['voice']  = voice
            metadata['path']   = path

        rsp.stream_to_file(path)

        rsp = Result(rsp)
        return [rsp], metadata

    def prepare(self, argument):
        assert 'voice' in argument.kwargs, "TTS requires voice selection."
        assert 'path' in argument.kwargs, "TTS requires path selection."
        voice       = str(argument.kwargs['voice']).lower()
        audio_file  = str(argument.kwargs['path'])
        prompt      = str(argument.prop.prompt)
        argument.prop.processed_input = (voice, audio_file, prompt)
