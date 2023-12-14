import logging
from typing import List

import requests
import tiktoken
from bardapi import Bard

from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG


class BardEngine(Engine, OpenAIMixin):
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

    def id(self) -> str:
        if   self.config['NEUROSYMBOLIC_ENGINE_MODEL'] and \
            (self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('bard')):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_remaining_tokens(self, prompts: list) -> int:
        # iterate over prompts and compute number of tokens
        prompt = prompts[0]
        val = len(self.tokenizer.encode(prompt))
        return int((self.max_tokens - val) * 0.98)

    def forward(self, argument):
        if self.bard is None:
            self.init_session()

        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input
        prompts_            = prompts_ if isinstance(prompts_, list) else [prompts_]
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            res = self.bard.get_answer(prompts_[0])
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.connection.root.predict
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        rsp    = [res['content']]
        output = rsp if isinstance(prompts_, list) else rsp[0]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            argument.prop.prepared_input = argument.prop.processed_input
            return

        user:   str = ""
        system: str = ""
        system      = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{payload}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"[INSTRUCTION]\n{val}"

        suffix: str = str(argument.prop.processed_input)
        if '=>' in suffix:
            user += f"[LAST TASK]\n"

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            for p in parts[:-1]:
                system += f"{p}\n"
            # last part is the user input
            suffix = parts[-1]
        user += f"{suffix}"

        if argument.prop.template_suffix is not None:
            user += f"\n[[PLACEHOLDER]]\n{argument.prop.template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        argument.prop.prepared_input = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
