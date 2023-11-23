import logging
from typing import Iterable, List

import torch
import openai
from tqdm import tqdm

from ..formatter import WhisperTimestampsFormatter
from .base import Engine
from .settings import SYMAI_CONFIG

from pathlib import Path
from openai import OpenAI


class TTSEngine(Engine):
    def __init__(self):
        super().__init__()
        config = SYMAI_CONFIG
        self.api_key  = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model_id = config['TEXT_TO_SPEECH_ENGINE_MODEL']
        self.tokens = []
        self.text = []
        self.client = OpenAI(api_key=self.api_key)

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'TEXT_TO_SPEECH_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['TEXT_TO_SPEECH_ENGINE_MODEL']

    def forward(self, **kwargs) -> List[str]:
        prompt = str(kwargs['prompt'])
        voice  = str(kwargs['voice']).lower()
        path  = str(kwargs['path'])

        input_handler   = kwargs.get("input_handler")
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

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        assert 'voice' in wrp_params, "TTS requires voice selection."
        assert 'path' in wrp_params, "TTS requires path selection."
        voice = str(wrp_params['voice'])
        audio_file = str(wrp_params['path'])
        wrp_params['voice'] = voice
        wrp_params['path'] = audio_file
