from typing import List
import torch
import requests
from PIL import Image
from transformers import BlipForConditionalGeneration, Blip2Processor
from .base import Engine
from .settings import SYMAI_CONFIG


class Blip2Engine(Engine):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # lazy loading
        self.processor = None  # lazy loading

        config = SYMAI_CONFIG
        self.model_id = config['CAPTION_ENGINE_MODEL']
        self.old_model_id = config['CAPTION_ENGINE_MODEL']

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'CAPTION_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['CAPTION_ENGINE_MODEL']

    def prepare(self, args, kwargs, wrp_params):
        pass

    def forward(self, *args, **kwargs) -> List[str]:
        if self.model is None:
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id, low_cpu_mem_usage=True).to(self.device)

        img, prompt = kwargs['image'], kwargs['prompt']

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((img, prompt))

        if 'http' in img:
            img = Image.open(requests.get(img, stream=True).raw).convert('RGB')
        elif '/' in img or '\\' in img:
            img = Image.open(img).convert('RGB')

        try:
            inputs = self.processor(img, prompt, return_tensors="pt").to(self.device)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.model.generate(**inputs)
            res = self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            if except_remedy is None:
                raise e
            res = except_remedy(e, prompt, *args, **kwargs)

        # remove the input text from the response
        res = res[len(prompt):].strip()

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(outputs)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input'] = (img, prompt)
            metadata['output'] = outputs

        return [res], metadata
