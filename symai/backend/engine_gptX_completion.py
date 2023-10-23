import logging
from typing import List

import openai
import tiktoken

from .base import Engine
from .mixin.openai import OpenAIMixin
from .settings import SYMAI_CONFIG


class GPTXCompletionEngine(Engine, OpenAIMixin):
    def __init__(self):
        super().__init__()
        config          = SYMAI_CONFIG
        openai.api_key  = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model      = config['NEUROSYMBOLIC_ENGINE_MODEL']
        logger          = logging.getLogger('openai')
        self.tokenizer  = tiktoken.encoding_for_model(self.model)
        self.pricing    = self.api_pricing()
        self.max_tokens = self.api_max_tokens()
        logger.setLevel(logging.WARNING)

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_remaining_tokens(self, prompts: list) -> int:
        # iterate over prompts and compute number of tokens
        prompt = prompts[0]
        val = len(self.tokenizer.encode(prompt, disallowed_special=()))
        return int((self.max_tokens - val) * 0.98)

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_            = prompts if isinstance(prompts, list) else [prompts]
        input_handler       = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        # send prompt to GPT-3
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts_)
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        suffix              = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            res = openai.Completion.create(model=model,
                                           prompt=prompts_,
                                           suffix=suffix,
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
            callback = openai.ChatCompletion.create
            kwargs['model'] = model
            res = except_remedy(e, prompts_, callback, self, *args, **kwargs)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompts_
            metadata['output'] = res

        rsp    = [r['text'] for r in res['choices']]
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

        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            user += f"[LAST TASK]\n"
        if wrp_params['prompt'] is not None:
            user += f"[INSTRUCTION]\n{str(wrp_params['prompt'])}"
        user += f"{suffix}"

        template_suffix = wrp_params['template_suffix'] if 'template_suffix' in wrp_params else None
        if template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        wrp_params['prompts'] = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
