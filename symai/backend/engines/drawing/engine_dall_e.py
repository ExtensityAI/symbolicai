import logging
import openai
import requests
import tempfile

from typing import Optional

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class DalleResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        # unpack the result
        if hasattr(value, 'data') and len(value.data) > 0:
            self._value = value.data[0].url

    # use tempfile to download the image
    def download_images(self):
        if not hasattr(self.value, 'data') or len(self.value.data) <= 0:
            return None
        files = []
        # download the images
        for url in enumerate(self.value.data):
            path = tempfile.NamedTemporaryFile(suffix='.png').name
            r = requests.get(url, allow_redirects=True)
            with open(path, 'wb') as f:
                f.write(r.content)
            files.append(path)
        return files


class DrawingEngine(Engine):
    def __init__(self, size: int = 1024, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        openai.api_key = self.config['DRAWING_ENGINE_API_KEY'] if api_key is None else api_key
        self.model = self.config['DRAWING_ENGINE_MODEL'] if model is None else model
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.size = size # defaults to 1024 instead of 512 because dall-e-3 requires 1024

    def id(self) -> str:
        if  self.config['DRAWING_ENGINE_API_KEY'] and self.config['DRAWING_ENGINE_MODEL'].startswith("dall-e"):
            return 'drawing'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'DRAWING_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['DRAWING_ENGINE_API_KEY']
        if 'DRAWING_ENGINE_MODEL' in kwargs:
            self.model = kwargs['DRAWING_ENGINE_MODEL']

    def forward(self, argument):
        prompt          = argument.prop.prepared_input
        kwargs          = argument.kwargs
        model           = kwargs.get('model', self.model)
        size            = f"{kwargs['image_size']}x{kwargs['image_size']}" if 'image_size' in kwargs else f"{self.size}x{self.size}"
        n               = kwargs.get('n', 1)
        quality         = kwargs.get('quality', 'standard')
        response_format = kwargs.get('response_format', 'url')
        style           = kwargs.get('style', 'vivid')
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        callback = None
        try:
            if kwargs['operation'] == 'create':

                callback = openai.images.generate
                res = openai.images.generate(
                    model=model,
                    prompt=prompt,
                    n=n,
                    size=size,
                    quality=quality,
                    response_format=response_format,
                    style=style,
                )
            elif kwargs['operation'] == 'variation':
                assert 'image_path' in kwargs
                image_path = kwargs['image_path']

                callback = openai.images.create_variation
                res = openai.images.create_variation(
                    model=model,
                    image=open(image_path, "rb"),
                    n=n,
                    size=size,
                    response_format=response_format,
                )
            elif kwargs['operation'] == 'edit':
                assert 'mask_path' in kwargs
                assert 'image_path' in kwargs
                mask_path = kwargs['mask_path']
                image_path = kwargs['image_path']

                callback = openai.images.edit
                res = openai.images.edit(
                    model=model,
                    image=open(image_path, "rb"),
                    mask=open(mask_path, "rb"),
                    prompt=prompt,
                    n=n,
                    size=size,
                    response_format=response_format,
                )
            else:
                raise Exception(f"Unknown operation: {kwargs['operation']}")

        except Exception as e:
            if except_remedy is None:
                raise e
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        rsp = DalleResult(res)
        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
