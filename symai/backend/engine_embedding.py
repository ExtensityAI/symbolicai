import logging
from typing import List

import openai

from .base import Engine
from .mixin.openai import OpenAIMixin
from .settings import SYMAI_CONFIG


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class EmbeddingEngine(Engine, OpenAIMixin):
    def __init__(self):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        config                  = SYMAI_CONFIG
        openai.api_key          = config['EMBEDDING_ENGINE_API_KEY']
        self.model              = config['EMBEDDING_ENGINE_MODEL']
        self.pricing            = self.api_pricing()
        self.max_tokens         = self.api_max_tokens()

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'EMBEDDING_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['EMBEDDING_ENGINE_API_KEY']
        if 'EMBEDDING_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['EMBEDDING_ENGINE_MODEL']

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_      = prompts if isinstance(prompts, list) else [prompts]
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None

        if input_handler:
            input_handler((prompts_,))

        try:
            res = openai.embeddings.create(model=self.model,
                                           input=prompts_)
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = openai.embeddings.create
            res = except_remedy(e, prompts_, callback, self, *args, **kwargs)

        rsp = [r.embedding for r in res.data]
        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompts_
            metadata['output'] = res
            metadata['model']  = self.model

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompts'] = wrp_params['entries']
