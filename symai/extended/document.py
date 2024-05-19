import os

from typing import Callable, Optional

from ..symbol import Expression, Symbol
from ..components import FileReader, Indexer
from ..formatter import ParagraphFormatter


class DocumentRetriever(Expression):
    def __init__(
            self,
            index_name: str = Indexer.DEFAULT,
            file = None,
            top_k = 5,
            formatter: Callable = ParagraphFormatter(),
            overwrite: bool = False,
            with_metadata: bool = False,
            raw_result: Optional[bool] = False,
            new_dim: Optional[int] = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        indexer = Indexer(index_name=index_name, top_k=top_k, formatter=formatter, auto_add=False, new_dim=new_dim)
        text    = None

        if not indexer.exists() or overwrite:
            indexer.register()
            if type(file) is str:
                file_path = file
                reader = FileReader()
                text = reader(file_path, with_metadata=with_metadata, **kwargs)
            else:
                text = str(file)

            self.index = indexer(
                    data=text, #@NOTE: we write the text to the index
                    raw_result=raw_result,
                    **kwargs
                )
        else:
            self.index = indexer(
                    raw_result=raw_result,
                    **kwargs
                    )

        self.text = Symbol(text)
        if text is not None:
            # save in home directory
            path = os.path.join(os.path.expanduser('~'), '.symai', 'temp', index_name)
            # create the directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)
            self.dump(os.path.join(path, 'dump_file'), replace=True)

    def forward(
            self,
            query: Symbol,
            raw_result: Optional[bool] = False,
        ) -> Symbol:
        return self.index(
                query,
                raw_result=raw_result,
                )

    def dump(self, path: str, replace: bool = True) -> Symbol:
        if self.text is None:
            raise ValueError('No text to save.')
        # save the text to a file
        self.text.save(path, replace=replace)
