from typing import List

import requests
import torch

try:
    from lavis.models import load_model_and_preprocess
except ImportError:
    load_model_and_preprocess = None
    print('Blip2 is not installed. Please install it with `pip install symbolicai[blip2]`')

from PIL import Image

from ...base import Engine
from ...settings import SYMAI_CONFIG


class Blip2Engine(Engine):
    def __init__(self):
        super().__init__()
        config              = SYMAI_CONFIG
        ids                 = config['CAPTION_ENGINE_MODEL'].split('/')
        self.name_id        = ids[0]
        self.model_id       = ids[1]
        self.model          = None  # lazy loading
        self.vis_processors = None  # lazy loading
        self.txt_processors = None  # lazy loading
        self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def id(self) -> str:
        return 'imagecaptioning'

    def command(self, argument):
        super().command(argument.kwargs)
        if 'CAPTION_ENGINE_MODEL' in argument.kwargs:
            self.model_id = argument.kwargs['CAPTION_ENGINE_MODEL']

    def forward(self, argument):
        if self.model is None:
            self.model, self.vis_processors, self.txt_processors  = load_model_and_preprocess(name       = self.name_id,
                                                                                              model_type = self.model_id,
                                                                                              is_eval    = True,
                                                                                              device     = self.device)

        image, prompt = argument.prop.prepared_input
        kwargs        = argument.kwargs
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        if 'http' in image:
            image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
        elif '/' in image or '\\' in image:
            image = Image.open(image).convert('RGB')

        try:
            image   = self.vis_processors['eval'](image).unsqueeze(0).to(self.device)
            prompt  = self.txt_processors['eval'](prompt)
            res     = self.model.generate(samples={"image": image, "prompt": prompt}, use_nucleus_sampling=True, num_captions=3)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.model.generate
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        return [res], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "Blip2Engine does not support processed_input."
        argument.prop.prepared_input = (argument.prop.image, argument.prop.prompt)