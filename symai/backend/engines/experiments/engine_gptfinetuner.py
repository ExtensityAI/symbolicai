import os
from typing import List

import openai

from ....utils import Args
from ...base import Engine
from ...settings import SYMAI_CONFIG


class GPTFineTuner(Engine):
    def __init__(self):
        super().__init__()
        config          = SYMAI_CONFIG
        openai.api_key  = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.base_model = "babbage"

    def forward(self, *args, **kwargs) -> List[str]:
        assert '__cmd__' in kwargs, "Missing __cmd__ argument"
        rsp = None

        raise NotImplementedError("GPTFineTuner is not implemented yet")

        del kwargs['__cmd__']

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = args
            metadata['output'] = rsp
            metadata['model']  = self.base_model

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass
