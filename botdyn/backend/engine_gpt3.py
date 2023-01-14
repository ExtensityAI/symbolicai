from .settings import BOTDYN_CONFIG
from typing import List
from .base import Engine
from time import sleep
import openai
import logging


class GPT3Engine(Engine):
    def __init__(self, max_retry: int = 3, api_cooldown_delay: int = 3):
        super().__init__()
        config = BOTDYN_CONFIG
        openai.api_key = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = config['NEUROSYMBOLIC_ENGINE_MODEL']
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.max_retry = max_retry
        self.api_cooldown_delay = api_cooldown_delay

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_ = prompts if isinstance(prompts, list) else [prompts]
        # send prompt to GPT-3
        stop = kwargs['stop'] if 'stop' in kwargs else None
        model = kwargs['model'] if 'model' in kwargs else self.model
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
        suffix = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))
        
        retry: int = 0
        success: bool = False
        errors: List[Exception] = []
        
        max_retry = kwargs['max_retry'] if 'max_retry' in kwargs else self.max_retry
        while not success and retry < max_retry:
            try:
                res = openai.Completion.create(model=model,
                                               prompt=prompts_,
                                               suffix=suffix,
                                               max_tokens=max_tokens,
                                               temperature=0.7,
                                               frequency_penalty=0,
                                               presence_penalty=0,
                                               top_p=1,
                                               n=1,
                                               stop=stop)
                output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
                if output_handler:
                    output_handler(res)
                success = True
            except Exception as e:                  
                errors.append(e)
                self.logger.warn(f"GPT-3 service is unavailable or caused an error. Retry triggered: {e}")
                sleep(self.api_cooldown_delay) # API cooldown
                # remedy for GPT-3 API limitation
                from botdyn.symbol import Symbol
                sym = Symbol(e)
                # TODO: fix with better prior length computation of max_tokens limit
                try:
                    if sym.contains('maximum context length', max_retry=0):
                        token_size = sym.extract("tokens in your prompt", max_retry=0)
                        token_size = token_size.cast(int)
                        max_tokens = 4097 - token_size
                        self.logger.warn(f"Try to remedy the exceeding of the maximum token limitation! Set max_tokens to {max_tokens}, tokens in prompt {token_size}")
                except Exception as e:
                    errors.append(e)
                    self.logger.warn(f"Failed to remedy the exceeding of the maximum token limitation! {e}")
            retry += 1
        
        if not success:
            msg = f"Failed to query GPT-3 after {max_retry} retries. Errors: {errors}"
            # interpret error
            from botdyn.symbol import Symbol
            sym = Symbol(errors)
            sym.analyze(exception=errors[-1], query="Explain the issue in this error message", max_retry=0)
            msg_reply = f"{msg}\n Analysis: {sym}"
            raise Exception(msg_reply)
        
        rsp = [r['text'] for r in res['choices']]      
        return rsp if isinstance(prompts, list) else rsp[0]
    
    def prepare(self, args, kwargs, wrp_params):
        ref = wrp_params['wrp_self']
        prompt: str = ''
        if 'static_context' in dir(ref) and len(ref.static_context) > 0:
            prompt += f"General Context:\n{ref.static_context}\n\n----------------\n\n"
        if 'dynamic_context' in wrp_params and len(str(wrp_params['dynamic_context'])) > 0:
            prompt += f"General Context:\n{str(wrp_params['dynamic_context'])}\n\n----------------\n\n"
        if wrp_params['prompt']:
            prompt += wrp_params['prompt']
        examples: List[str] = wrp_params['examples']
        suffix: str = wrp_params['suffix']
        # build query
        message = f'{prompt}\n' if prompt and len(prompt) > 0 else ''
        if examples:
            message += f"Examples:\n"
            message += f"{str(examples)}\n"
            message += "\nLast Task:\n"
        message += f"{suffix}"
        wrp_params['prompts'] = [message]
