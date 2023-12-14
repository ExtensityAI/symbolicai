import requests

from typing import Optional

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
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Opening JSON file
        self.config = SYMAI_CONFIG
        self.headers = {
            "apikey": self.config['OCR_ENGINE_API_KEY'] if api_key is None else api_key
        }

    def id(self) -> str:
        if self.config['OCR_ENGINE_API_KEY']:
            return 'ocr'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'OCR_ENGINE_API_KEY' in kwargs:
            self.headers = {
                "apikey": kwargs['OCR_ENGINE_API_KEY']
            }

    def forward(self, argument):
        kwargs    = argument.kwargs
        image_url = argument.prop.image
        url       = f"https://api.apilayer.com/image_to_text/url?url={image_url}"
        payload   = {}
        response    = requests.request("GET", url, headers=self.headers, data = payload)
        status_code = response.status_code
        rsp         = response.text

        metadata  = {}

        rsp       = ApiLayerResult(response.text, status_code)
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "OCREngine does not support processed_input."
        image  = str(argument.prop.image)
        argument.prop.prepared_input = image

