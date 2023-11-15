from typing import Callable, Optional
import os

from .. import Expression, FileReader, Indexer, ParagraphFormatter, Symbol


class DocumentRetriever(Expression):
    def __init__(self, file = None, index_name: str = Indexer.DEFAULT, top_k = 5, formatter: Callable = ParagraphFormatter(), overwrite: bool = False, **kwargs):
        super().__init__()
        indexer = Indexer(index_name=index_name, top_k=top_k, formatter=formatter, auto_add=False)
        text = None
        if not indexer.exists() or overwrite:
            indexer.register()
            if type(file) is str:
                file_path = file
                reader = FileReader()
                text = reader(file_path, **kwargs)
            else:
                text = str(file)
            self.index = indexer(text, **kwargs)
        else:
            self.index = indexer(**kwargs)

        self.text = Symbol(text)
        if text is not None:
            # save in home directory
            path = os.path.join(os.path.expanduser('~'), '.symai', 'temp', index_name)
            # create the directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)
            self.dump(os.path.join(path, 'dump_file'), replace=True)

    def forward(self, query: Optional[Symbol], raw_result: bool = False) -> Symbol:
        return self.index(query, raw_result=raw_result)

    def dump(self, path: str, replace: bool = True) -> Symbol:
        if self.text is None:
            raise ValueError('No text to save.')
        # save the text to a file
        self.text.save(path, replace=replace)
