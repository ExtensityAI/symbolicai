import logging
import time
import requests

from typing import Optional

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class FluxResult(Result):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        # unpack the result
        if value.get('status') == "Ready":
            self._value = value.get('result').get('sample')


class DrawingEngine(Engine):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config['DRAWING_ENGINE_API_KEY'] if api_key is None else api_key
        self.model = self.config['DRAWING_ENGINE_MODEL'] if model is None else model
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)

    def id(self) -> str:
        if  self.config['DRAWING_ENGINE_API_KEY'] and self.config['DRAWING_ENGINE_MODEL'].startswith("flux"):
            return 'drawing'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'DRAWING_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['DRAWING_ENGINE_API_KEY']
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
            'steps': steps,
            'prompt_upsampling': prompt_upsampling,
            'seed': seed,
            'guidance': guidance,
            'safety_tolerance': safety_tolerance,
            'interval': interval,
            'output_format': output_format,
        }

        if kwargs.get('operation') == 'create':
            try:
                response = requests.post(
                    f'https://api.bfl.ml/v1/{self.model}',
                    headers=headers,
                    json=payload
                ).json()

                request_id = response.get("id")
                if not request_id:
                    raise Exception("Failed to get request ID!")

                while True:
                    time.sleep(5)
                    result = requests.get(
                        'https://api.bfl.ml/v1/get_result',
                        headers=headers,
                        params={'id': request_id}
                    ).json()

                    if result["status"] == "Ready":
                        rsp = FluxResult(result)
                        break
                    else:
                        raise Exception(f"Failed to get result: {result}")

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
