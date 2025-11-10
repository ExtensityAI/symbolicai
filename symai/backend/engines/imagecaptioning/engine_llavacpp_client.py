import io
import json
import logging
from pathlib import Path

import requests
from PIL.Image import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG


def image_to_byte_array(image: Image, format='PNG') -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format=format)
  # Turn the BytesIO object back into a bytes object
  return imgByteArr.getvalue()


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
        self.config = SYMAI_CONFIG
        self.host = host
        self.port = port
        self.timeout = timeout
        self.name = self.__class__.__name__

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
            with Path(image['content']).open('rb') as f:
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
        except_remedy = kwargs.get('except_remedy')
        try:
            # use http localhost 8000 to send a request to the server
            rsp = requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = rsp.text
        except Exception as e:
            if except_remedy is None:
                raise e
            def callback():
                return requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        res    = LLaMAResult(res)
        rsp    = [res]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def _handle_raw_input(self, argument) -> bool:
        if not argument.prop.raw_input:
            return False
        if not argument.prop.processed_input:
            UserMessage('Need to provide a prompt instruction to the engine if raw_input is enabled.', raise_with=ValueError)
        argument.prop.prepared_input = argument.prop.processed_input
        return True

    def _append_context_sections(self, system: str, argument) -> str:
        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"[ADDITIONAL CONTEXT]\n{payload!s}\n\n"

        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{examples!s}\n\n"

        return system

    def _build_user_instruction(self, argument) -> str:
        user = ""
        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            user += f"[INSTRUCTION]\n{val}"
        return user

    def _extract_system_instructions(self, argument, system: str, suffix: str) -> tuple[str, str]:
        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            consumed = 0
            for part in parts:
                if 'SYSTEM_INSTRUCTION' in part:
                    system += f"{part}\n"
                    consumed += 1
                else:
                    break
            suffix = '\n>>>\n'.join(parts[consumed:])
        return system, suffix

    def _append_template_suffix(self, user: str, argument) -> str:
        if argument.prop.template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{argument.prop.template_suffix!s}\n\n"
            user += "Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"
        return user

    def prepare(self, argument):
        if self._handle_raw_input(argument):
            return

        system: str = ""
        system = f'{system}\n' if system and len(system) > 0 else ''
        system = self._append_context_sections(system, argument)

        user = self._build_user_instruction(argument)
        suffix: str = str(argument.prop.processed_input)
        system, suffix = self._extract_system_instructions(argument, system, suffix)
        user += f"{suffix}"
        user = self._append_template_suffix(user, argument)

        user_prompt = { "role": "user", "content": user }
        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
            { "role": "image", "content": argument.prop.image }
        ]
