from typing import Optional

from .. import Expression, FileReader, Indexer, Symbol


class DocumentRetriever(Expression):
    def __init__(self, file_path: str, index_name: str = Indexer.DEFAULT):
        super().__init__()
        reader = FileReader()
        indexer = Indexer(index_name=index_name)
        text = reader(file_path)
        self.index = indexer(text)

    def forward(self, query: Optional[Symbol]) -> Symbol:
        return self.index(query)
