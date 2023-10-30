from typing import Callable, Optional

from .. import Expression, FileReader, Indexer, ParagraphFormatter, Symbol
from ..strategy import InvalidRequestErrorRemedyStrategy


class DocumentRetriever(Expression):
    def __init__(self, file_path: str, index_name: str = Indexer.DEFAULT, top_k = 5, formatter: Callable = ParagraphFormatter(), overwrite: bool = False, **kwargs):
        super().__init__()
        self.remedy = InvalidRequestErrorRemedyStrategy()
        indexer = Indexer(index_name=index_name, top_k=top_k, formatter=formatter)
        if not indexer.exists() or overwrite:
            reader = FileReader()
            text = reader(file_path, **kwargs)
            self.index = indexer(text, **kwargs)

    def forward(self, query: Optional[Symbol]) -> Symbol:
        return self.index(query, except_remedy=self.remedy)
