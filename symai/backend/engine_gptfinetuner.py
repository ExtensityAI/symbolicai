from typing import List

from .base import Engine
from .settings import SYMAI_CONFIG


class GPTFineTuner(Engine):
    def __init__(self):
        super().__init__()
        self.model        = None # lazy loading
        self.preprocessor = None # lazy loading

        config = SYMAI_CONFIG
        self.model_id = config['FINETUNING_ENGINE_MODEL']
        self.old_model_id = config['FINETUNING_ENGINE_MODEL']

        self.processor = None

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'FINETUNING_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['FINETUNING_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:
        # TODO: implement this

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (image_url, text)
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass

