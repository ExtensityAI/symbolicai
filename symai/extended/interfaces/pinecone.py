from ...backend.engines.index.engine_pinecone import PineconeIndexEngine, PineconeResult
from ...symbol import Expression
from ...utils import UserMessage


class pinecone(Expression):
    def __init__(self, index_name = PineconeIndexEngine._default_index_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_name = index_name
        self.name = self.__class__.__name__

    def __call__(self, stmt: str, operation: str = "search", index_name = None, **kwargs) -> PineconeResult:
        stmt  = self._to_symbol(stmt)
        index = self.index_name if index_name is None else index_name
        if   operation == "search":
            return self.get(query=stmt.embedding, index_name=index, ori_query=stmt.value, **kwargs)
        if operation == "add":
            return self.add(doc=stmt.zip(), index_name=index, **kwargs)
        if operation == "config":
            return self.index(path=stmt.value, index_name=index, **kwargs)
        UserMessage("Operation not supported", raise_with=NotImplementedError)
        return None
