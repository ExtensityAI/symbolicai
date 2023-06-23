import logging
from time import sleep
from typing import List

import openai
import tiktoken

from .base import Engine
from .mixin.openai import OpenAIMixin
from .settings import SYMAI_CONFIG


class GPTXChatEngine(Engine, OpenAIMixin):
    def __init__(self):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        config                  = SYMAI_CONFIG
        openai.api_key          = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model              = config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.tokenizer          = tiktoken.encoding_for_model(self.model)
        self.pricing            = self.api_pricing()

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_            = prompts
        input_handler       = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            res = openai.ChatCompletion.create(model=model,
                                                messages=prompts_,
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                top_p=top_p,
                                                stop=stop,
                                                n=1)
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if except_remedy is None:
                raise e
            rsp = except_remedy(e, prompts_, *args, **kwargs)

        rsp = [r['message']['content'] for r in res['choices']]
        return rsp if isinstance(prompts, list) else rsp[0]

    def prepare(self, args, kwargs, wrp_params):
        _non_verbose_output = """[META INSTRUCTIONS START]\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n"""
        user: str = ""
        system: str = ""

        system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

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
        if examples:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            user += f"[LAST TASK]\n"
        if wrp_params['prompt'] is not None:
            user += str(wrp_params['prompt'])
        user += f"{suffix}"

        template_suffix = wrp_params['template_suffix'] if 'template_suffix' in wrp_params else None
        if template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        wrp_params['prompts'] = [
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ]

