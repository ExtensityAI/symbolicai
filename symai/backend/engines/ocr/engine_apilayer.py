import requests

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


class ApiLayerResult(Result):
    def __init__(self, text, status_code=200):
        super().__init__()
        self.raw = text
        try:
            dict_ = self._to_symbol(text).ast()
            self._value = dict_['all_text'] if 'all_text' in dict_ else f'OCR Engine Error: {text} - status code {status_code}'
        except:
            self._value = f'OCR Engine Error: {text} - status code {status_code}'


class OCREngine(Engine):
    def __init__(self):
        super().__init__()
        # Opening JSON file
        self.config = SYMAI_CONFIG
        self.headers = {
            "apikey": self.config['OCR_ENGINE_API_KEY']
        }

    def id(self) -> str:
        if self.config['OCR_ENGINE_API_KEY']:
            return 'ocr'
        return super().id() # default to unregistered

    def command(self, argument):
        super().command(argument.kwargs)
        if 'OCR_ENGINE_API_KEY' in argument.kwargs:
            self.headers = {
                "apikey": argument.kwargs['OCR_ENGINE_API_KEY']
            }

    def forward(self, argument):
        kwargs    = argument.kwargs
        image_url = argument.prop.image
        url       = f"https://api.apilayer.com/image_to_text/url?url={image_url}"
        payload   = {}

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((url, payload))

        response    = requests.request("GET", url, headers=self.headers, data = payload)
        status_code = response.status_code
        rsp         = response.text
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata  = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (image_url, url)
            metadata['output'] = response

        rsp       = ApiLayerResult(response.text, status_code)
        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.processed_input = argument.prop.image

