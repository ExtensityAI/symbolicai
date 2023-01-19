from typing import List
from .base import Engine
# importing required modules
import PyPDF2


class FileEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        path = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((path,))
        
        if 'pdf' in path:
            rsp = ''
            with open(path, 'rb') as f:
                # creating a pdf reader object
                pdf_reader = PyPDF2.PdfReader(f)
                n_pages = len(pdf_reader.pages)
                if 'n_pages' in kwargs:
                    n_pages = kwargs['n_pages']
                for i in range(n_pages):
                    page = pdf_reader.pages[i]
                    rsp += page.extract_text()
        else:
            with open(path, 'r') as f:
                rsp = f.read()
                
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        pass