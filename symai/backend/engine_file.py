from typing import List

import PyPDF2
from tika import unpack

from .base import Engine


class FileEngine(Engine):
    def __init__(self):
        super().__init__()

    def reset_eof_of_pdf_return_stream(self, pdf_stream_in: list):
        actual_line = len(pdf_stream_in)  # Predefined value in case EOF not found
        # find the line position of the EOF
        for i, x in enumerate(pdf_stream_in[::-1]):
            if b'%%EOF' in x:
                actual_line = len(pdf_stream_in)-i
                print(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
                break

        # return the list up to that point
        return pdf_stream_in[:actual_line]

    def fix_pdf(self, file_path: str):
        # opens the file for reading
        with open(file_path, 'rb') as p:
            txt = (p.readlines())

        # get the new list terminating correctly
        txtx = self.reset_eof_of_pdf_return_stream(txt)

        # write to new pdf
        new_file_path = f'{file_path}_fixed.pdf'
        with open(new_file_path, 'wb') as f:
            f.writelines(txtx)

        fixed_pdf = PyPDF2.PdfReader(new_file_path)
        return fixed_pdf

    def read_text(self, pdf_reader, range_):
        txt = ''
        n_pages = len(pdf_reader.pages)
        if range_ is None:
            for i in range(n_pages):
                page = pdf_reader.pages[i]
                txt += page.extract_text()
        else:
            for i in range(n_pages)[range_]:
                page = pdf_reader.pages[i]
                txt += page.extract_text()
        return txt

    def forward(self, *args, **kwargs) -> List[str]:
        path          = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((path,))

        range_ = None
        if 'range' in kwargs:
            range_ = kwargs['range']
            if isinstance(range_, tuple) or isinstance(range_, list):
                range_ = slice(*range_)

        if '.pdf' in path:
            rsp = ''
            try:
                with open(str(path), 'rb') as f:
                    # creating a pdf reader object
                    pdf_reader = PyPDF2.PdfReader(f)
                    rsp = self.read_text(pdf_reader, range_)
            except Exception as e:
                print(f'Error reading PDF: {e} | {path}')
                if 'fix_pdf' not in kwargs or not kwargs['fix_pdf']:
                    raise e
                fixed_pdf = self.fix_pdf(str(path))
                pdf_reader_fixed = PyPDF2.PdfReader(fixed_pdf)
                rsp = self.read_text(pdf_reader_fixed, range_)
        else:
            try:
                file_ = unpack.from_file(str(path))
                if 'content' in file_:
                    rsp = file_['content']
                else:
                    rsp = str(file_)
            except Exception as e:
                print(f'Error reading file: {e} | {path}')
                raise e

        # ensure encoding is utf8
        rsp = rsp.encode('utf8', 'ignore').decode('utf8', 'ignore')

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = path
            metadata['output'] = rsp
            metadata['range']  = range_
            metadata['fix_pdf'] = kwargs['fix_pdf'] if 'fix_pdf' in kwargs else False

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass
