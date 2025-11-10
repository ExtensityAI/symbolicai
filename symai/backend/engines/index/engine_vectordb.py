import itertools
from copy import deepcopy
from typing import ClassVar

from ....extended.vectordb import VectorDB
from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


class VectorDBResult(Result):
    def __init__(self, res, query: str, embedding: list, **kwargs):
        super().__init__(res, **kwargs)
        self.raw = res
        self._query = query
        self._value = self._process(res)
        self._metadata.raw = embedding

    def _process(self, res):
        if not res:
            return None
        try:
            res = self._to_symbol(res).ast()
        except Exception as e:
            message = ['Sorry, failed to interact with index. Please check index name and try again later:', str(e)]
            # package the message for the IndexResult class
            res = {'matches': [{'metadata': {'text': '\n'.join(message)}}]}
        return [v['metadata']['text'] for v in res]

    def _unpack_matches(self):
        if not self.value:
            return
        for i, match in enumerate(self.value):
            match_value = match.strip()
            if match_value.startswith('# ----[FILE_START]') and '# ----[FILE_END]' in match_value:
                m = match_value.split('[FILE_CONTENT]:')[-1].strip()
                splits = m.split('# ----[FILE_END]')
                assert len(splits) >= 2, f'Invalid file format: {splits}'
                content = splits[0]
                file_name = ','.join(splits[1:]) # TODO: check why there are multiple file names
                yield file_name.strip(), content.strip()
            else:
                yield i+1, match_value

    def __str__(self):
        str_view = ''
        for filename, content in self._unpack_matches():
            # indent each line of the content
            content_view = '\n'.join(['  ' + line for line in content.split('\n')])
            str_view += f'* {filename}\n{content_view}\n\n'
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


class VectorDBIndexEngine(Engine):
    # Updated default values to be congruent with VectorDB's defaults
    _default_index_name = 'dataindex'
    _default_index_dims = 768
    _default_index_top_k = 5
    _default_index_metric = 'cosine'
    _index_dict: ClassVar[dict[str, object]] = {}
    _index_storage_file: ClassVar[str | None] = None
    def __init__(
            self,
            index_name=_default_index_name,
            index_dims=_default_index_dims,
            index_top_k=_default_index_top_k,
            index_metric=_default_index_metric,
            index_dict=_index_dict,
            index_storage_file=_index_storage_file,
            **_kwargs
        ):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.index_name = index_name
        self.index_dims = index_dims
        self.index_top_k = index_top_k
        self.index_metric = index_metric
        self.storage_file = index_storage_file
        self.model = None
        # Initialize an instance of VectorDB
        # Note that embedding_function and vectors are not passed as VectorDB will compute it on the fly
        self.index = index_dict
        self.name = self.__class__.__name__

    def id(self) -> str:
        if not self.config['INDEXING_ENGINE_API_KEY'] or self.config['INDEXING_ENGINE_API_KEY'] == '':
            return 'index'
        return super().id() # default to unregistered

    def forward(self, argument):
        query = argument.prop.prepared_input
        operation = argument.prop.operation
        index_name = argument.prop.index_name or self.index_name
        index_dims = argument.prop.index_dims or self.index_dims
        top_k = argument.prop.top_k or self.index_top_k
        metric = argument.prop.metric or self.index_metric
        similarities = argument.prop.similarities or False
        storage_file = argument.prop.storage_file or self.storage_file
        kwargs = argument.kwargs
        rsp = None

        self._init(index_name, top_k, index_dims, metric)

        if operation == 'search':
            if isinstance(query, list) and len(query) > 1:
                UserMessage('VectorDB indexing engine does not support multiple queries. Pass a single string query instead.', raise_with=ValueError)
            query_vector = self.index[index_name].embedding_function([query])[0]
            results = self.index[index_name](vector=query_vector, top_k=top_k, return_similarities=similarities)
            rsp = [{'metadata': {'text': result}} for result in results]
        elif operation == 'add':
            assert isinstance(query, list), 'VectorDB indexing requires a list of queries at insertion, even if there is only one query.'
            documents = []
            vectors = []
            for q in query:
                vectors.append(self.index[index_name].embedding_function([q])[0])
                documents.append(q)
            self.index[index_name].add(documents=documents, vectors=vectors)
        elif operation == 'config':
            assert kwargs, 'Please provide a configuration by passing the appropriate kwargs. Currently, only `load`, `save`, `purge`.'
            maybe_as_prompt = kwargs.get('prompt')
            if kwargs.get('load', maybe_as_prompt == 'load'):
                assert storage_file, 'Please provide a `storage_file` path to load the pre-computed index.'
                self.load(index_name, storage_file, index_dims, top_k, metric)
            elif kwargs.get('save', maybe_as_prompt == 'save'):
                self.save(index_name, storage_file)
            elif kwargs.get('purge', maybe_as_prompt == 'purge'):
                self.purge(index_name)
            else:
                UserMessage('Invalid configuration; please use either "load", "save", or "purge".', raise_with=ValueError)
        else:
            UserMessage('Invalid operation; please use either "search", "add", or "config".', raise_with=ValueError)

        metadata = {}
        rsp = VectorDBResult(rsp, query[0], None)

        return [rsp], metadata

    def _init(self, index_name, top_k, index_dims, metric, embedding_model=not None):
        if index_name not in self.index:
            self.index[index_name] = VectorDB(
                index_name=index_name,
                index_dims=index_dims,
                top_k=top_k,
                similarity_metric=metric,
                embedding_model=embedding_model #@NOTE: the VectorDBIndexEngine class uses precomputed embeddings so the model is not needed in the VectorDB class
            )

    def prepare(self, argument):
        assert not argument.prop.processed_input, 'VectorDB indexing engine does not support processed_input.'
        argument.prop.prepared_input = argument.prop.prompt
        argument.prop.limit = 1

    def load(self, index_name, storage_file, index_dims, top_k, metric):
        self.index[index_name] = VectorDB(
            index_dims=index_dims,
            top_k=top_k,
            similarity_metric=metric,
            load_on_init=storage_file,
            index_name=index_name,
        )

    def save(self, index_name = None, storage_file = None):
        index_name = index_name or self.index_name
        storage_file = storage_file or self._index_storage_file
        self.index[index_name].save(storage_file)

    def purge(self, index_name = None):
        index_name = index_name or self.index_name
        self.index[index_name].purge(index_name)
