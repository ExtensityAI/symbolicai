import itertools
import warnings
import numpy as np

warnings.filterwarnings('ignore', module='pinecone')
try:
    import pinecone
except:
    pinecone = None

from ...base import Engine
from ...settings import SYMAI_CONFIG
from .... import core_ext
from ....symbol import Result


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


class PineconeResult(Result):
    def __init__(self, res, query: str, embedding: list, **kwargs):
        super().__init__(res, **kwargs)
        self.raw                 = res
        self._query              = query
        self._value              = self._process(res)
        self._metadata.raw       = embedding

    def _process(self, res):
        if not res:
            return None
        try:
            res = self._to_symbol(res).ast()
        except Exception as e:
            message = ['Sorry, failed to interact with index. Please check index name and try again later:', str(e)]
            # package the message for the IndexResult class
            res = {'matches': [{'metadata': {'text': '\n'.join(message)}}]}
        return [v['metadata']['text'] for v in res['matches']]

    def _unpack_matches(self):
        if not self.value:
            return

        for i, match in enumerate(self.value):
            match = match.strip()
            if match.startswith('# ----[FILE_START]') and '# ----[FILE_END]' in match:
                m = match.split('[FILE_CONTENT]:')[-1].strip()
                splits = m.split('# ----[FILE_END]')
                assert len(splits) >= 2, 'Invalid file format: {}'.format(splits)
                content = splits[0]
                file_name = ','.join(splits[1:]) # TODO: check why there are multiple file names
                yield file_name.strip(), content.strip()
            else:
                yield i+1, match

    def __str__(self):
        str_view = ''
        for filename, content in self._unpack_matches():
            # indent each line of the content
            content = '\n'.join(['  ' + line for line in content.split('\n')])
            str_view += f'* {filename}\n{content}\n\n'
        return f'''
[RESULT]
{'-=-' * 13}

Query: {self._query}

{'-=-' * 13}

Matches:

{str_view}
{'-=-' * 13}
'''

    def _repr_html_(self) -> str:
        # return a nicely styled HTML list results based on retrieved documents
        doc_str = ''
        for filename, content in self._unpack_matches():
            doc_str += f'<li><a href="{filename}"><b>{filename}</a></b><br>{content}</li>\n'
        return f'<ul>{doc_str}</ul>'


class PineconeIndexEngine(Engine):
    _default_api_key         = SYMAI_CONFIG['INDEXING_ENGINE_API_KEY']
    _default_environment     = SYMAI_CONFIG['INDEXING_ENGINE_ENVIRONMENT']
    _default_index_name      = 'dataindex'
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
        self.index          = None

    def id(self) -> str:
        if SYMAI_CONFIG['INDEXING_ENGINE_API_KEY']:
            if pinecone is None:
                print('Pinecone is not installed. Please install it with `pip install symbolicai[pinecone]`.')
            return 'index'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'INDEXING_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['INDEXING_ENGINE_API_KEY']
        if 'INDEXING_ENGINE_ENVIRONMENT' in kwargs:
            self.environment = kwargs['INDEXING_ENGINE_ENVIRONMENT']

    def _configure_index(self, **kwargs):
        index_name = kwargs['index_name'] if 'index_name' in kwargs else self.index_name

        del_ = kwargs['index_del'] if 'index_del' in kwargs else False
        if self.index is not None and del_:
            pinecone.delete_index(index_name)

        get_ = kwargs['index_get'] if 'index_get' in kwargs else False
        if self.index is not None and get_:
            self.index = pinecone.Index(index_name=index_name)

    def forward(self, argument):
        assert self.api_key,        'Please set the API key for Pinecone indexing engine.'
        assert self.environment,    'Please set the environment for Pinecone indexing engine.'
        assert self.index_name,     'Please set the index name for Pinecone indexing engine.'

        kwargs        = argument.kwargs
        embedding     = argument.prop.prepared_input
        query         = argument.prop.ori_query
        operation     = argument.prop.operation
        index_name    = argument.prop.index_name if argument.prop.index_name else self.index_name
        rsp           = None

        if self.index is None:
            self._init_index_engine(index_name=index_name, index_dims=self.index_dims, index_metric=self.index_metric)

        if index_name != self.index_name:
            assert index_name, 'Please set a valid index name for Pinecone indexing engine.'
            # switch index
            self.index_name     = index_name
            kwargs['index_get'] = True
            self._configure_index(**kwargs)

        if operation == 'search':
            index_top_k    = kwargs['index_top_k'] if 'index_top_k' in kwargs else self.index_top_k
            index_values   = kwargs['index_values'] if 'index_values' in kwargs else self.index_values
            index_metadata = kwargs['index_metadata'] if 'index_metadata' in kwargs else self.index_metadata
            rsp = self._query(embedding, index_top_k, index_values, index_metadata)

        elif operation == 'add':
            for ids_vectors_chunk in chunks(embedding, batch_size=100):
                self._upsert(ids_vectors_chunk)

        elif operation == 'config':
            self._configure_index(**kwargs)

        else:
            raise ValueError('Invalid operation')

        metadata = {}

        rsp = PineconeResult(rsp, query, embedding)
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, 'Pinecone indexing engine does not support processed_input.'
        argument.prop.prepared_input = argument.prop.prompt

    def _init_index_engine(self, index_name, index_dims, index_metric):
        pinecone.init(api_key=self.api_key, environment=self.environment)

        if index_name is not None and index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=index_dims, metric=index_metric)

        self.index = pinecone.Index(index_name=index_name)

    def _upsert(self, vectors):
        @core_ext.retry(tries=self.tries, delay=self.delay, max_delay=self.max_delay, backoff=self.backoff, jitter=self.jitter)
        def _func():
            return self.index.upsert(vectors=vectors)

        return _func()

    def _query(self, query, index_top_k, index_values, index_metadata):
        @core_ext.retry(tries=self.tries, delay=self.delay, max_delay=self.max_delay, backoff=self.backoff, jitter=self.jitter)
        def _func():
            return self.index.query(vector=query, top_k=index_top_k, include_values=index_values, include_metadata=index_metadata)

        return _func()
