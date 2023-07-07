import itertools
import warnings
from typing import List

warnings.filterwarnings('ignore', module='pinecone')
try:
    import pinecone
except:
    pinecone = None

from ..core import retry
from .base import Engine
from .settings import SYMAI_CONFIG


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


class IndexEngine(Engine):
    _default_api_key         = SYMAI_CONFIG['INDEXING_ENGINE_API_KEY']
    _default_environment     = SYMAI_CONFIG['INDEXING_ENGINE_ENVIRONMENT']
    _default_index_name      = 'data-index'
    _default_index_dims      = 1536
    _default_index_top_k     = 5
    _default_index_metric    = 'cosine'
    _default_index_values    = True
    _default_index_metadata  = True
    _default_retry_tries     = 20
    _default_retry_delay     = 0.5
    _default_retry_max_delay = -1
    _default_retry_backoff   = 1
    _default_retry_jitter    = 0

    def __init__(
            self,
            api_key=_default_api_key,
            environment=_default_environment,
            index_name=_default_index_name,
            index_dims=_default_index_dims,
            index_top_k=_default_index_top_k,
            index_metric=_default_index_metric,
            index_values=_default_index_values,
            index_metadata=_default_index_metadata,
            tries=_default_retry_tries,
            delay=_default_retry_delay,
            max_delay=_default_retry_max_delay,
            backoff=_default_retry_backoff,
            jitter=_default_retry_jitter,
        ):
        super().__init__()
        self.index_name     = index_name
        self.index_dims     = index_dims
        self.index_top_k    = index_top_k
        self.index_values   = index_values
        self.index_metadata = index_metadata
        self.index_metric   = index_metric
        self.api_key        = api_key
        self.environment    = environment
        self.tries          = tries
        self.delay          = delay
        self.max_delay      = max_delay
        self.backoff        = backoff
        self.jitter         = jitter

        self._init_index_engine()

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'INDEXING_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['INDEXING_ENGINE_API_KEY']
        if 'INDEXING_ENGINE_ENVIRONMENT' in wrp_params:
            self.environment = wrp_params['INDEXING_ENGINE_ENVIRONMENT']

    def forward(self, *args, **kwargs) -> List[str]:
        operation     = kwargs['operation']
        query         = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        rsp           = None

        if input_handler:
            input_handler((query, ))

        if operation == 'search':
            index_top_k    = kwargs['index_top_k'] if 'index_top_k' in kwargs else self.index_top_k
            index_values   = kwargs['index_values'] if 'index_values' in kwargs else self.index_values
            index_metadata = kwargs['index_metadata'] if 'index_metadata' in kwargs else self.index_metadata

            rsp = self._query(query, index_top_k, index_values, index_metadata)

        elif operation == 'add':
            for ids_vectors_chunk in chunks(query, batch_size=100):
                rsp = self._upsert(ids_vectors_chunk)

        elif operation == 'config':
            index_name = kwargs['index_name'] if 'index_name' in kwargs else self.index_name

            del_ = kwargs['index_del'] if 'index_del' in kwargs else False
            if self.index is not None and del_:
                pinecone.delete_index(index_name)

            get_ = kwargs['index_get'] if 'index_get' in kwargs else False
            if self.index is not None and get_:
                self.index = pinecone.Index(index_name=index_name)

        else:
            raise ValueError('Invalid operation')

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (operation, query)
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompt'] = wrp_params['prompt']

    def _init_index_engine(self):
        pinecone.init(api_key=self.api_key, environment=self.environment)

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, dimension=self.index_dims, metric=self.index_metric)

        self.index = pinecone.Index(index_name=self.index_name)

    def _upsert(self, vectors):
        @retry(tries=self.tries, delay=self.delay, max_delay=self.max_delay, backoff=self.backoff, jitter=self.jitter)
        def _func():
            return self.index.upsert(vectors=vectors)

        return _func()

    def _query(self, query, index_top_k, index_values, index_metadata):
        @retry(tries=self.tries, delay=self.delay, max_delay=self.max_delay, backoff=self.backoff, jitter=self.jitter)
        def _func():
            return self.index.query(vector=query, top_k=index_top_k, include_values=index_values, include_metadata=index_metadata)

        return _func()
