import requests

from typing import List

from ...base import Engine
from ...settings import SYMAI_CONFIG


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

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'OCR_ENGINE_API_KEY' in wrp_params:
            self.headers = {
                "apikey": wrp_params['OCR_ENGINE_API_KEY']
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
        if status_code != 200:
            raise Exception(f"OCR request failed with status code {status_code}")

        rsp = response.text
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (image_url, url)
            metadata['output'] = response

        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.processed_input = argument.prop.image

