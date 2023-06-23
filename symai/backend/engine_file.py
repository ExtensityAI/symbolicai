from typing import List

import PyPDF2
from tika import unpack

from .base import Engine


class FileEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        path = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((path,))

        range_ = None
        if 'range' in kwargs:
            range_ = kwargs['range']
            if isinstance(range_, tuple) or isinstance(range_, list):
                range_ = slice(*range_)

        if 'pdf' in path:
            rsp = ''
            with open(str(path), 'rb') as f:
                # creating a pdf reader object
                pdf_reader = PyPDF2.PdfReader(f)
                n_pages = len(pdf_reader.pages)
                if range_ is None:
                    for i in range(n_pages):
                        page = pdf_reader.pages[i]
                        rsp += page.extract_text()
                else:
                    for i in range(n_pages)[range_]:
                        page = pdf_reader.pages[i]
                        rsp += page.extract_text()
        else:
            rsp = unpack.from_file(str(path))['content']

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        return [rsp]

    def prepare(self, args, kwargs, wrp_params):
        pass
