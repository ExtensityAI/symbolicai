import logging
import numpy as np
import openai

from typing import Optional

from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class EmbeddingEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.config        = SYMAI_CONFIG
        openai.api_key     = self.config['EMBEDDING_ENGINE_API_KEY'] if api_key is None else api_key
        self.model         = self.config['EMBEDDING_ENGINE_MODEL'] if model is None else model
        self.pricing       = self.api_pricing()
        self.max_tokens    = self.api_max_tokens()
        self.embedding_dim = self.api_embedding_dims()

    def id(self) -> str:
        if  self.config['EMBEDDING_ENGINE_API_KEY']:
            return 'embedding'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'EMBEDDING_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['EMBEDDING_ENGINE_API_KEY']
        if 'EMBEDDING_ENGINE_MODEL' in kwargs:
            self.model = kwargs['EMBEDDING_ENGINE_MODEL']

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        args           = argument.args
        kwargs         = argument.kwargs

        input_        = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        except_remedy = kwargs.get('except_remedy')
        new_dim       = kwargs.get('new_dim')

        try:
            res = openai.embeddings.create(model=self.model,
                                           input=input_)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = openai.embeddings.create
            res = except_remedy(e, input_, callback, self, *args, **kwargs)

        if new_dim:
            mn = min(new_dim, self.embedding_dim) #@NOTE: new_dim should be less than or equal to the original embedding dim
            rsp = [self._normalize_l2(r.embedding[:mn]) for r in res.data]
        else:
            rsp = [r.embedding for r in res.data]

        metadata = {}
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "EmbeddingEngine does not support processed_input."
        argument.prop.prepared_input = argument.prop.entries

    def _normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm).tolist()
