import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from ..components import FileReader, Indexer
from ..formatter import ParagraphFormatter
from ..symbol import Expression, Symbol


class DocumentRetriever(Expression):
    def __init__(
            self,
            source: Optional[str] = None,
            *,
            index_name: str = Indexer.DEFAULT,
            top_k: int = 5,
            max_depth: int = 1,
            formatter: Callable = ParagraphFormatter(),
            overwrite: bool = False,
            with_metadata: bool = False,
            raw_result: Optional[bool] = False,
            new_dim: Optional[int] = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.indexer = Indexer(index_name=index_name, top_k=top_k, formatter=formatter, auto_add=False, new_dim=new_dim)
        self.reader  = FileReader(with_metadata=with_metadata)
        self.new_dim = new_dim

        if overwrite:
            self.config(None, purge=True, index_name=self.indexer.index_name, **kwargs)

        # we insert the text into the index if (1) index does not exist and (2) there's a specific source
        if source is not None and not self.indexer.exists():
            self.indexer.register()
            text = self.parse_source(source, with_metadata=with_metadata, max_depth=max_depth, **kwargs)
            self.index = self.indexer(data=text, raw_result=raw_result, **kwargs)
        else:
            # we don't insert the text at initialization since the index already exists and there's no specific source
            self.index = self.indexer(raw_result=raw_result, **kwargs)

    def forward(
            self,
            query: Symbol,
            raw_result: Optional[bool] = False,
        ) -> Symbol:
        return self.index(
                query,
                raw_result=raw_result,
                )

    def insert(self, source: Union[str, Path], **kwargs):
        # dynamically insert data into the index given a session
        # the data can be:
        #  - a string (e.g. something that the user wants to insert)
        #  - a file path (e.g. a new file that the user wants to insert)
        #  - a directory path (e.g. a new directory that the user wants to insert)
        text = self.parse_source(source, with_metadata=kwargs.get('with_metadata', False), max_depth=kwargs.get('max_depth', 1), **kwargs)
        #NOTE: Do we need `new_dim` here?
        self.add(text, index_name=self.indexer.index_name, **kwargs)
        self.config(None, save=True, index_name=self.indexer.index_name, **kwargs)

    def parse_source(self, source: str, with_metadata: bool, max_depth: int, **kwargs) -> List[Union[str, 'TextContainer']]:
        maybe_path = Path(source)
        if isinstance(source, str) and not (maybe_path.is_file() or maybe_path.is_dir()):
            return Symbol(source).zip(new_dim=self.new_dim)
        if maybe_path.is_dir():
            files = FileReader.get_files(source, max_depth)
            return self.reader(files, with_metadata=with_metadata, **kwargs)
        if maybe_path.is_file():
            return self.reader(source, with_metadata=with_metadata, **kwargs)
        raise ValueError(f"Invalid source: {source}; must be a file, directory, or string")
