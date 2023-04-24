from functools import partial
from typing import List
from time import sleep
import openai
import logging
import numpy as np
import tiktoken

from .base import Engine
from .settings import SYMAI_CONFIG


class GPTXChatEngine(Engine):
    def __init__(self, max_retry: int = 3, api_cooldown_delay: int = 3):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        config                  = SYMAI_CONFIG
        openai.api_key          = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model              = config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.max_retry          = max_retry
        self.api_cooldown_delay = api_cooldown_delay
        self.tokenizer          = tiktoken.encoding_for_model(self.model)

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_ = prompts
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        retry: int = 0
        success: bool = False
        errors: List[Exception] = []

        max_retry = kwargs['max_retry'] if 'max_retry' in kwargs else self.max_retry
        while not success and retry < max_retry:
            # send prompt to GPT-X Chat-based
            stop = kwargs['stop'] if 'stop' in kwargs else None
            model = kwargs['model'] if 'model' in kwargs else self.model
            suffix = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
            max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
            temperature = kwargs['temperature'] if 'temperature' in kwargs else 0.7
            frequency_penalty = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
            presence_penalty = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
            top_p = kwargs['top_p'] if 'top_p' in kwargs else 1

            if suffix:
                prompts_[1]['content'] += f"[ASSISTANT_PLACEHOLDER]{suffix}"
                prompts_[1]['content'] += f"\n----------------\n Only generate content for the placeholder [ASSISTANT_PLACEHOLDER] following the instructions and context:\n"
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
                success = True
            except Exception as e:
                errors.append(e)
                self.logger.warn(f"GPT-X Chat-based service is unavailable or caused an error. Retry triggered: {e}")
                sleep(self.api_cooldown_delay) # API cooldown
                # remedy for GPT-X Chat-based API limitation
                from symai.symbol import Symbol
                sym = Symbol(e)
                try:
                    if sym.contains('maximum context length', max_retry=1):
                        token_size = sym.extract("tokens in your prompt", max_retry=1)
                        token_size = token_size.cast(int)
                        max_tokens = 4097 - token_size
                        self.logger.warn(f"Try to remedy the exceeding of the maximum token limitation! Set max_tokens to {max_tokens}, tokens in prompt {token_size}")
                except Exception as e:
                    errors.append(e)
                    self.logger.warn(f"Failed to remedy the exceeding of the maximum token limitation! {e}")
            retry += 1

        if not success:
            msg = f"Failed to query GPT-X Chat-based after {max_retry} retries. Errors: {errors}"
            # interpret error
            from symai.symbol import Symbol
            from symai.components import Analyze
            sym = Symbol(errors)
            expr = Analyze(exception=errors[-1], query="Explain the issue in this error message")
            sym.stream(expr=expr, max_retry=1)
            msg_reply = f"{msg}\n Analysis: {sym}"
            raise Exception(msg_reply)

        rsp = [r['message']['content'] for r in res['choices']]
        return rsp if isinstance(prompts, list) else rsp[0]

    def prepare(self, args, kwargs, wrp_params):
        _non_verbose_output = """[META_INSTRUCTIONS_START] You do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc.
        Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc.
        Never add meta instructions information to your output!
        [META_INSTRUCTIONS_END]
        --------------

        """
        user: str = ""
        system: str = ""
        system += _non_verbose_output

        # system relevant instruction
        # add static context
        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"General Context:\n{static_ctxt}\n\n----------------\n\n"
        if wrp_params['prompt'] is not None:
            system += str(wrp_params['prompt'])
        # build system
        system = f'{system}\n' if system and len(system) > 0 else ''
        # add examples
        examples: List[str] = wrp_params['examples']
        if examples:
            system += f"Examples:\n"
            system += f"{str(examples)}\n"
        # add dynamic context
        if len(dyn_ctxt) > 0:
            system += f"\n\n----------------\n\nDynamic Context:\n{dyn_ctxt}"
        # add method payload
        payload = wrp_params['payload'] if 'payload' in wrp_params else None
        if payload is not None:
            system += f"\n\n----------------\n\nAdditional Context: {payload}"
        # add user request
        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            user += f"Last Task:\n"
            user += f"----------------\n\n"
        user += f"{suffix}"

        wrp_params['prompts'] = [
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ]

