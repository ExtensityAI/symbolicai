from .settings import SYMAI_CONFIG
from typing import List
from .base import Engine
from time import sleep
import openai
import logging


class EmbeddingEngine(Engine):
    def __init__(self, max_retry: int = 3, api_cooldown_delay: int = 3):
        super().__init__()
        config = SYMAI_CONFIG
        openai.api_key = config['EMBEDDING_ENGINE_API_KEY']
        self.model = config['EMBEDDING_ENGINE_MODEL']
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.max_retry = max_retry
        self.api_cooldown_delay = api_cooldown_delay

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_ = prompts if isinstance(prompts, list) else [prompts]
        retry: int = 0
        success: bool = False
        errors: List[Exception] = []
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))
        
        max_retry = kwargs['max_retry'] if 'max_retry' in kwargs else self.max_retry
        while not success and retry < max_retry:
            try:
                res = openai.Embedding.create(model=self.model,
                                              input=prompts_)
                output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
                if output_handler:
                    output_handler(res)
                success = True
            except Exception as e:
                errors.append(e)
                self.logger.warn(f"GPT Embedding service is unavailable or caused an error. Retry triggered: {e}")
                sleep(self.api_cooldown_delay) # API cooldown
            retry += 1
        
        if not success:
            msg = f"Failed to query GPT Embedding after {max_retry} retries. Errors: {errors}"
            # interpret error
            from symai.symbol import Symbol
            from symai.components import Analyze
            sym = Symbol(errors)
            expr = Analyze(exception=errors[-1], query="Explain the issue in this error message")
            sym.stream(expr=expr, max_retry=1)
            msg_reply = f"{msg}\n Analysis: {sym}"
            raise Exception(msg_reply)
        
        rsp = [r['embedding'] for r in res['data']]      
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompts'] = wrp_params['entries']
