import logging
from typing import List

import openai

from .base import Engine
from .settings import SYMAI_CONFIG


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class ImageRenderingEngine(Engine):
    def __init__(self, size: int = 512):
        super().__init__()
        config = SYMAI_CONFIG
        openai.api_key = config['IMAGERENDERING_ENGINE_API_KEY']
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.size = size

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'IMAGERENDERING_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['IMAGERENDERING_ENGINE_API_KEY']

    def forward(self, prompt: str, *args, **kwargs) -> List[str]:
        size          = f"{kwargs['image_size']}x{kwargs['image_size']}" if 'image_size' in kwargs else f"{self.size}x{self.size}"
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        callback = None
        try:
            if kwargs['operation'] == 'create':
                input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                if input_handler:
                    input_handler((prompt,))

                callback = openai.images.generate
                res = openai.images.generate(
                    prompt=prompt,
                    n=1,
                    size=size
                )
            elif kwargs['operation'] == 'variation':
                assert 'image_path' in kwargs
                image_path = kwargs['image_path']

                input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                if input_handler:
                    input_handler((prompt, image_path))

                callback = openai.images.create_variation
                res = openai.images.create_variation(
                    image=open(image_path, "rb"),
                    n=1,
                    size=size
                )
            elif kwargs['operation'] == 'edit':
                assert 'mask_path' in kwargs
                assert 'image_path' in kwargs
                mask_path = kwargs['mask_path']
                image_path = kwargs['image_path']

                input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                if input_handler:
                    input_handler((prompt, image_path, mask_path))

                callback = openai.images.edit
                res = openai.images.edit(
                    image=open(image_path, "rb"),
                    mask=open(mask_path, "rb"),
                    prompt=prompt,
                    n=1,
                    size=size
                )
            else:
                raise Exception(f"Unknown operation: {kwargs['operation']}")

            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)

        except Exception as e:
            if except_remedy is None:
                raise e
            res = except_remedy(e, prompt, callback, *args, **kwargs)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompt
            metadata['output'] = res
            metadata['size']   = size

        rsp = res.data[0].url
        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        prompt = str(wrp_params['prompt'])
        prompt += wrp_params['processed_input']
        wrp_params['prompt'] = prompt
