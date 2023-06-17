import itertools
import warnings
from typing import List

warnings.filterwarnings('ignore', module='pinecone')
import pinecone

from .base import Engine
from .settings import SYMAI_CONFIG


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


class IndexEngine(Engine):
    def __init__(self):
        super().__init__()
        self.documents       = None
        self.index           = None
        self.index_name      = 'data-index'
        self.index_dims      = 1536
        self.index_top_k     = 5
        self.index_values    = True
        self.index_metadata  = True
        self.index_metric    = 'cosine'
        self.api_key         = SYMAI_CONFIG['INDEXING_ENGINE_API_KEY']
        self.environment     = SYMAI_CONFIG['INDEXING_ENGINE_ENVIRONMENT']
        self.old_api_key     = self.api_key
        self.old_environment = self.environment

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'INDEXING_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['INDEXING_ENGINE_API_KEY']
        if 'INDEXING_ENGINE_ENVIRONMENT' in wrp_params:
            self.environment = wrp_params['INDEXING_ENGINE_ENVIRONMENT']

    def forward(self, *args, **kwargs) -> List[str]:
        if self.documents is None or self.old_api_key != self.api_key or self.old_environment != self.environment:
            index_name   = kwargs['index_name'] if 'index_name' in kwargs else self.index_name
            index_dims   = kwargs['index_dims'] if 'index_dims' in kwargs else self.index_dims
            index_metric = kwargs['index_metric'] if 'index_metric' in kwargs else self.index_metric

            pinecone.init(api_key=self.api_key, environment=self.environment)
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=index_dims, metric=index_metric)

            self.index           = pinecone.Index(index_name=index_name)
            self.old_api_key     = self.api_key
            self.old_environment = self.environment

        operation     = kwargs['operation']
        query         = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((query, ))

        if operation == 'search':
            index_top_k    = kwargs['index_top_k'] if 'index_top_k' in kwargs else self.index_top_k
            index_values   = kwargs['index_values'] if 'index_values' in kwargs else self.index_values
            index_metadata = kwargs['index_metadata'] if 'index_metadata' in kwargs else self.index_metadata

            rsp = self.index.query(
                    vector=query,
                    top_k=index_top_k,
                    include_values=index_values,
                    include_metadata=index_metadata
            )

        elif operation == 'add':
            for ids_vectors_chunk in chunks(query, batch_size=100):
                rsp = self.index.upsert(vectors=ids_vectors_chunk)

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

        return [rsp]

    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompt'] = wrp_params['prompt']

