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

    def forward(self, argument):
        kwargs = argument.kwargs
        assert '__cmd__' in kwargs, "Missing __cmd__ argument"
        rsp = None

        raise NotImplementedError("GPTFineTuner is not implemented yet")

        del kwargs['__cmd__']

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "GPTFineTuner does not support processed_input."
