import logging
from typing import List

import requests
import tiktoken
from bardapi import Bard

from .base import Engine
from .mixin.openai import OpenAIMixin
from .settings import SYMAI_CONFIG


class GPTXCompletionEngine(Engine, OpenAIMixin):
    def __init__(self, timeout: int = 30):
        super().__init__()
        self.timeout    = timeout
        config          = SYMAI_CONFIG
        self.api_key    = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model      = config['NEUROSYMBOLIC_ENGINE_MODEL']
        logger          = logging.getLogger('openai')
        self.tokenizer  = tiktoken.encoding_for_model(self.model)
        self.pricing    = self.api_pricing()
        self.max_tokens = self.api_max_tokens()
        self.bard       = None
        logger.setLevel(logging.WARNING)

    def init_session(self):
        self.session    = requests.Session()
        self.session.headers = {
                    "Host": "bard.google.com",
                    "X-Same-Domain": "1",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "Origin": "https://bard.google.com",
                    "Referer": "https://bard.google.com/",
                }
        self.session.cookies.set("__Secure-1PSID", self.api_key)
        self.bard = Bard(token=self.api_key,
                         session=self.session,
                         timeout=self.timeout)

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_remaining_tokens(self, prompts: list) -> int:
        # iterate over prompts and compute number of tokens
        prompt = prompts[0]
        val = len(self.tokenizer.encode(prompt))
        return int((self.max_tokens - val) * 0.98)

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        if self.bard is None:
            self.init_session()

        prompts_            = prompts if isinstance(prompts, list) else [prompts]
        input_handler       = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            res = self.bard.get_answer(prompts_[0])
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if except_remedy is None:
                raise e
            res = except_remedy(e, prompts_, *args, **kwargs)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompts_
            metadata['output'] = res
            metadata['model']  = self.model
            metadata['max_tokens'] = self.max_tokens

        rsp    = [res['content']]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def prepare(self, args, kwargs, wrp_params):
        if 'raw_input' in wrp_params:
            wrp_params['prompts'] = wrp_params['raw_input']
            return

        user:   str = ""
        system: str = ""
        system      = f'{system}\n' if system and len(system) > 0 else ''

        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = wrp_params['payload'] if 'payload' in wrp_params else None
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{payload}\n\n"

        examples: List[str] = wrp_params['examples']
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if wrp_params['prompt'] is not None and len(wrp_params['prompt']) > 0 and ']: <<<' not in str(wrp_params['processed_input']): # TODO: fix chat hack
            user += f"[INSTRUCTION]\n{str(wrp_params['prompt'])}"

        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            user += f"[LAST TASK]\n"

        parse_system_instructions = False if 'parse_system_instructions' not in wrp_params else wrp_params['parse_system_instructions']
        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            for p in parts[:-1]:
                system += f"{p}\n"
            # last part is the user input
            suffix = parts[-1]
        user += f"{suffix}"

        template_suffix = wrp_params['template_suffix'] if 'template_suffix' in wrp_params else None
        if template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        wrp_params['prompts'] = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
