import logging
import requests
import json
import io

from typing import List
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL.Image import Image

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result


def image_to_byte_array(image: Image, format='PNG') -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format=format)
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


class LLaMAResult(Result):
    def __init__(self, value=None, *args, **kwargs):
        super().__init__(value, *args, **kwargs)
        self._value = value
        self.error  = None
        self.raw    = value
        self._perse_result()

    def _perse_result(self):
        val        = json.loads(self.value)
        self.value = val
        if 'error' in val:
            self.error = val['error']
        if 'content' in val:
            self.value = val['content']


class LLaMACppClientEngine(Engine):
    def __init__(self, host: str = 'localhost', port: int = 8080, timeout: int = 240):
        super().__init__()
        logger = logging.getLogger('nesy_client')
        logger.setLevel(logging.WARNING)
        self.config     = SYMAI_CONFIG
        self.host       = host
        self.port       = port
        self.timeout    = timeout

    def id(self) -> str:
        if  self.config['CAPTION_ENGINE_MODEL'] and \
            'llavacpp' in self.config['CAPTION_ENGINE_MODEL']:
            return 'imagecaptioning'
        return super().id() # default to unregistered

    @property
    def max_tokens(self):
        return 4096

    def forward(self, argument):
        prompts             = argument.prop.prepared_input
        kwargs              = argument.kwargs
        system, user, image = prompts
        # escape special characters
        system              = system['content']
        user                = user['content']

        if isinstance(image['content'], Image):
            # format image to bytes
            format_ = argument.prop.image_format if argument.prop.image_format else 'PNG'
            im_bytes = image_to_byte_array(image['content'], format=format_)
        else:
            # Convert image to bytes, open as binary
            with open(image['content'], 'rb') as f:
                im_bytes = f.read()
        # Create multipart/form-data payload
        payload      = MultipartEncoder(
            fields={
                'user_prompt': ('user_prompt', user, 'text/plain'),
                'image_file': ('image_file', im_bytes, 'application/octet-stream'),
                'system_prompt': ('system_prompt', system, 'text/plain')
            }
        )
        # Update the headers for multipart/form-data
        headers       = {'Content-Type': payload.content_type}
        api           = f'http://{self.host}:{self.port}/llava'
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None
        try:
            # use http localhost 8000 to send a request to the server
            rsp = requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = rsp.text
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = lambda: requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        res    = LLaMAResult(res)
        rsp    = [res]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            argument.prop.prepared_input = argument.prop.processed_input
            return

        user:   str = ""
        system: str = ""
        system = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"[ADDITIONAL CONTEXT]\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            # in this engine, instructions are considered as user prompts
            user += f"[INSTRUCTION]\n{val}"

        suffix: str = str(argument.prop.processed_input)

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            c = 0
            for i, p in enumerate(parts):
                if 'SYSTEM_INSTRUCTION' in p:
                    system += f"{p}\n"
                    c += 1
                else:
                    break
            # last part is the user input
            suffix = '\n>>>\n'.join(parts[c:])
        user += f"{suffix}"

        if argument.prop.template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{str(argument.prop.template_suffix)}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        user_prompt = { "role": "user", "content": user }
        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
            { "role": "image", "content": argument.prop.image }
        ]
