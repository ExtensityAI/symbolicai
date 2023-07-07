from typing import List

import requests
from PIL import Image
from transformers import Blip2Model, Blip2Processor

from .base import Engine
from .settings import SYMAI_CONFIG


class Blip2Engine(Engine):
    def __init__(self):
        super().__init__()
        self.model =  None # lazy loading
        self.preprocessor = None # lazy loading

        config = SYMAI_CONFIG
        self.model_id = config['VISION_ENGINE_MODEL']
        self.old_model_id = config['VISION_ENGINE_MODEL']

        self.processor = None

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'VISION_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['VISION_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(device)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (image_url, text)
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass

