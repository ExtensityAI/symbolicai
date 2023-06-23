import logging
from time import sleep
from typing import List

import openai

from .base import Engine
from .settings import SYMAI_CONFIG


class GPT3Engine(Engine):
    def __init__(self):
        super().__init__()
        config = SYMAI_CONFIG
        openai.api_key = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = config['NEUROSYMBOLIC_ENGINE_MODEL']
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_            = prompts if isinstance(prompts, list) else [prompts]
        input_handler       = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        # send prompt to GPT-3
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        suffix              = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
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
            rsp = except_remedy(e, prompts_, *args, **kwargs)

        rsp = [r['text'] for r in res['choices']]
        return rsp if isinstance(prompts, list) else rsp[0]

    def prepare(self, args, kwargs, wrp_params):
        prompt: str = ''
        # add static context
        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            prompt += f"General Context:\n{static_ctxt}\n\n----------------\n\n"
        if wrp_params['prompt'] is not None:
            prompt += str(wrp_params['prompt'])
        # build operation
        message = f'{prompt}\n' if prompt and len(prompt) > 0 else ''
        # add examples
        examples: List[str] = wrp_params['examples']
        if examples:
            message += f"Examples:\n"
            message += f"{str(examples)}\n"
        # add dynamic context
        if len(dyn_ctxt) > 0:
            message += f"\n\n----------------\n\nDynamic Context:\n{dyn_ctxt}"
        # add method payload
        payload = wrp_params['payload'] if 'payload' in wrp_params else None
        if payload is not None:
            message += f"\n\n----------------\n\nAdditional Context: {payload}"

        # add user request
        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            message += f"Last Task:\n"
            message += f"----------------\n\n"
        message += f"{suffix}"
        wrp_params['prompts'] = [message]
