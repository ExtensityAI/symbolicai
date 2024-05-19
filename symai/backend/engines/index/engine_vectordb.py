import itertools

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result
from ....extended.vectordb import VectorDB


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
        return [v['metadata']['text'] for v in res]

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


class VectorDBIndexEngine(Engine):
    # Updated default values to be congruent with VectorDB's defaults
    _default_index_name      = 'dataindex'
    _default_index_dims      = 768
    _default_index_top_k     = 5
    _default_index_metric    = 'cosine'
    _index_dict              = {}
    _index_storage_file      = None

    def __init__(
            self,
            index_name=_default_index_name,
            index_dims=_default_index_dims,
            index_top_k=_default_index_top_k,
            index_metric=_default_index_metric,
            index_dict=_index_dict,
            index_storage_file=_index_storage_file,
            **kwargs
        ):
        super().__init__()
        self.index_name     = index_name
        self.index_dims     = index_dims
        self.index_top_k    = index_top_k
        self.index_metric   = index_metric
        self.storage_file   = index_storage_file
        self.model          = None
        # Initialize an instance of VectorDB
        # Note that embedding_function and vectors are not passed as VectorDB will compute it on the fly
        self.index = index_dict

    def id(self) -> str:
        if not SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'] or SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'] == '':
            return 'index'
        return super().id() # default to unregistered

    def forward(self, argument):
        embedding    = argument.prop.prepared_input
        query        = argument.prop.ori_query
        operation    = argument.prop.operation
        index_name   = argument.prop.index_name or self.index_name
        index_dims   = argument.prop.index_dims or self.index_dims
        top_k        = argument.prop.top_k or self.index_top_k
        metric       = argument.prop.metric or self.index_metric
        kwargs       = argument.kwargs
        similarities = argument.prop.similarities or False
        storage_file = argument.prop.storage_file or self.storage_file
        rsp          = None

        self._init(index_name, top_k, index_dims, metric)

        # Process each operation
        if operation == 'search':
            # Get the query text from the prepared input
            query_vector = embedding[0]
            # Perform search in the VectorDB
            results = self.index[index_name](vector=query_vector, top_k=top_k, return_similarities=similarities)
            rsp = [{'metadata': {'text': result}} for result in results]

        elif operation == 'add':
            # Add provided documents (and optionally vectors) to the VectorDB
            documents = []
            vectors = []
            for uid, vector, document in embedding:
                vectors.append(vector)
                documents.append(document['text'])
            self.index[index_name].add(documents=documents, vectors=vectors)

        elif operation == 'config':
            # Handle any configurations if needed (not applicable for in-memory VectorDB)
            assert kwargs, 'Please provide a configuration dictionary.'
            # Check if the index is to be persisted or loaded
            if argument.prop.load:
                assert storage_file, 'Please provide a `storage_file` path to load the pre-computed index.'
                # Load the pre-computed index from the provided path
                self.index[index_name] = VectorDB(
                    index_dims=index_dims,
                    top_k=top_k,
                    similarity_metric=metric,
                    load_on_init=storage_file,
                    index_name=index_name
                )
            else:
                # Save the pre-computed index to the provided path
                self.index[index_name].save(storage_file)

        else:
            raise ValueError('Invalid operation; please use either "search", "add", or "config".')

        metadata = {}

        rsp = VectorDBResult(rsp, query, None)
        return [rsp], metadata

    def _init(self, index_name, top_k, index_dims, metric, embedding_model=not None):
        # Initialize the VectorDB if not already initialized
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

    def save(self, index_name = None, storage_file = None):
        index_name   = index_name or self.index_name
        storage_file = storage_file or self._index_storage_file
        # Save the pre-computed index to the provided path
        self.index[index_name].save(storage_file)

    def purge(self, index_name = None):
        index_name   = index_name or self.index_name
        # Purge the pre-computed index from the database path
        self.index[index_name].purge(index_name)
