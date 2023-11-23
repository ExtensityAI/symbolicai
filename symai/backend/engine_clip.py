from typing import List

import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import Engine
from .settings import SYMAI_CONFIG


class CLIPEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model =  None # lazy loading
        self.preprocessor = None # lazy loading

        config = SYMAI_CONFIG
        self.model_id = config['VISION_ENGINE_MODEL']
        self.old_model_id = config['VISION_ENGINE_MODEL']

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'VISION_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['VISION_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:
        if self.model is None or self.model_id != self.old_model_id:
            self.model        = CLIPModel.from_pretrained(self.model_id)
            self.processor    = CLIPProcessor.from_pretrained(self.model_id)
            self.old_model_id = self.model_id

        image_url = kwargs['image'] if 'image' in kwargs else None
        text      = kwargs['prompt']

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((image_url, text))

        if text is None:
            image = Image.open(requests.get(image_url, stream=True).raw)
            inputs = self.processor(images=image, return_tensors="pt")
            rsp = self.model.get_image_features(**inputs)
        elif image_url is None:
            inputs = self.processor(text=text, return_tensors="pt")
            rsp = self.model.get_text_features(**inputs)
        else:
            image = Image.open(requests.get(image_url, stream=True).raw)
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            rsp = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        rsp = rsp.detach().cpu().numpy()

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (image_url, text)
            metadata['output'] = rsp
            metadata['model']  = self.model_id

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass

