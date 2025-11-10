from ...backend.engines.index.engine_vectordb import VectorDBIndexEngine, VectorDBResult
from ...symbol import Expression
from ...utils import UserMessage


class naive_vectordb(Expression):
    def __init__(self, index_name = VectorDBIndexEngine._default_index_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_name = index_name
        self.name = self.__class__.__name__

    def __call__(
        self,
        query: str | list[str],
        operation: str = "search",
        index_name: str | None = None,
        storage_file: str | None = None,
        **kwargs
    ) -> VectorDBResult:
        index = self.index_name if index_name is None else index_name
        if operation == "search":
            return self.get(query=query, index_name=index, **kwargs)
        if operation == "add":
            if isinstance(query, list):
                for q in query:
                    self.add(doc=[q], index_name=index, **kwargs)
                return None
            self.add(doc=[query], index_name=index, **kwargs)
            return None
        if operation == "config":
            self.config(path=query, index_name=index, storage_file=storage_file, **kwargs)
            return None
        UserMessage(f"Operation not supported: {operation}", raise_with=NotImplementedError)
        return None
