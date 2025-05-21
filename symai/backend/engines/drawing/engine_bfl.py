import logging
import tempfile
import time
from typing import Optional

import requests

from ....symbol import Result
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class FluxResult(Result):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        # unpack the result
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        url = value.get('result').get('sample')
        request = requests.get(url, allow_redirects=True)
        request.raise_for_status()
        with open(path, "wb") as f:
            f.write(request.content)
        self._value = [path]


class DrawingEngine(Engine):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config['DRAWING_ENGINE_API_KEY'] if api_key is None else api_key
        self.model = self.config['DRAWING_ENGINE_MODEL'] if model is None else model
        self.name = self.__class__.__name__

    def id(self) -> str:
        if  self.config['DRAWING_ENGINE_API_KEY'] and self.config['DRAWING_ENGINE_MODEL'].startswith("flux"):
            return 'drawing'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'DRAWING_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['DRAWING_ENGINE_API_KEY']
        if 'DRAWING_ENGINE_MODEL' in kwargs:
            self.model = kwargs['DRAWING_ENGINE_MODEL']

    def forward(self, argument):
        prompt = argument.prop.prepared_input
        kwargs = argument.kwargs
        model = kwargs.get('model', self.model)
        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 768)
        steps = kwargs.get('steps', 40)
        prompt_upsampling = kwargs.get('prompt_upsampling', False)
        seed = kwargs.get('seed', None)
        guidance = kwargs.get('guidance', None)
        safety_tolerance = kwargs.get('safety_tolerance', 2)
        interval = kwargs.get('interval', None)
        output_format = kwargs.get('output_format', 'png')
        except_remedy = kwargs.get('except_remedy', None)

        headers = {
            'accept': 'application/json',
            'x-key': self.api_key,
            'Content-Type': 'application/json',
        }

        payload = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'num_inference_steps': steps,
            'guidance_scale': guidance,
            'seed': seed,
            'safety_tolerance': safety_tolerance,
        }
        # drop any None values so Flux API won't return 500
        payload = {k: v for k, v in payload.items() if v is not None}

        if kwargs.get('operation') == 'create':
            try:
                response = requests.post(
                    f'https://api.us1.bfl.ai/v1/{self.model}',
                    headers=headers,
                    json=payload
                )
                # fail early on HTTP errors
                response.raise_for_status()
                data = response.json()
                request_id = data.get("id")
                if not request_id:
                    raise Exception(f"Failed to get request ID! Response payload: {data}")

                while True:
                    time.sleep(5)

                    result = requests.get(
                        'https://api.us1.bfl.ai/v1/get_result',
                        headers=headers,
                        params={'id': request_id}
                    )

                    result.raise_for_status()
                    result = result.json()

                    if result["status"] == "Ready":
                        rsp = FluxResult(result)
                        break

            except Exception as e:
                if except_remedy is None:
                    raise e
                rsp = except_remedy(self, e, None, argument)

            metadata = {}
            return [rsp], metadata
        else:
            raise Exception(f"Unknown operation: {kwargs['operation']}")

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
