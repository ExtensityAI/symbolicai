from ... import core
from ...symbol import Expression
from ...backend.engines.index.engine_vectordb import VectorDBResult, VectorDBIndexEngine


class vectordb(Expression):
    def __init__(self, index_name = VectorDBIndexEngine._default_index_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_name = index_name

    def __call__(self, stmt: str, operation: str = "search", index_name = None, **kwargs) -> VectorDBResult:
        stmt  = self._to_symbol(stmt)
        index = self.index_name if index_name is None else index_name
        if   operation == "search":
            return self.get(query=stmt.embedding, index_name=index, ori_query=stmt.value, **kwargs)
        elif operation == "add":
            return self.add(doc=stmt.zip(), index_name=index, **kwargs)
        elif operation == "config":
            return self.index(path=stmt.value, index_name=index, **kwargs)
        raise NotImplementedError("Operation not supported")
