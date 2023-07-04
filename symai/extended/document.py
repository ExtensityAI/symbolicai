from typing import Optional

from .. import Expression, FileReader, Indexer, Symbol


class DocumentRetriever(Expression):
    def __init__(self, file_path: str):
        super().__init__()
        reader = FileReader()
        indexer = Indexer()
        text = reader(file_path)
        self.index = indexer(text)

    def forward(self, query: Optional[Symbol]) -> Symbol:
        return self.index(query)
