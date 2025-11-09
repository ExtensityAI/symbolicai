import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pypdf
import tika
from tika import unpack

from ....utils import CustomUserWarning
from ...base import Engine

# Initialize Tika lazily to avoid spawning JVMs prematurely for all workers
_TIKA_STATE = {"initialized": False}

def _ensure_tika_vm():
    if not _TIKA_STATE["initialized"]:
        with contextlib.suppress(Exception):
            tika.initVM()
        logging.getLogger('tika').setLevel(logging.CRITICAL)
        _TIKA_STATE["initialized"] = True


@dataclass
class TextContainer:
    id: str
    page: str
    text: str


class FileEngine(Engine):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def id(self) -> str:
        return 'files'

    def _read_slice_file(self, file_path, argument):
        # check if file is empty
        with_metadata = argument.kwargs.get('with_metadata', False)
        id            = Path(argument.prop.prepared_input).stem.replace(' ', '_')
        if file_path is None or file_path.strip() == '':
            return None

        # check if file slice is used
        slices_ = None
        if '[' in file_path and ']' in file_path:
            file_parts = file_path.split('[')
            file_path = file_parts[0]
            # remove string up to '[' and after ']'
            slices_s = file_parts[1].split(']')[0].split(',')
            slices_ = []
            for s in slices_s:
                if s == '':
                    continue
                if ':' in s:
                    s_split = s.split(':')
                    if len(s_split) == 2:
                        start_slice = int(s_split[0]) if s_split[0] != '' else None
                        end_slice = int(s_split[1]) if s_split[1] != '' else None
                        slices_.append(slice(start_slice, end_slice, None))
                    elif len(s_split) == 3:
                        start_slice = int(s_split[0]) if s_split[0] != '' else None
                        end_slice = int(s_split[1]) if s_split[1] != '' else None
                        step_slice = int(s_split[2]) if s_split[2] != '' else None
                        slices_.append(slice(start_slice, end_slice, step_slice))
                else:
                    slices_.append(int(s))

        path_obj = Path(file_path)

        # check if file exists
        assert path_obj.exists(), f'File does not exist: {file_path}'

        # verify if file is empty
        if path_obj.stat().st_size <= 0:
            return ''

        # For common plain-text extensions, avoid Tika overhead
        ext = path_obj.suffix.lower()
        if ext in {'.txt', '.md', '.py', '.json', '.yaml', '.yml', '.csv', '.tsv', '.log'}:
            try:
                with path_obj.open(encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content is None:
                    return None
                # Apply slicing by lines, mirroring the Tika branch
                lines = content.split('\n')
                if slices_ is not None:
                    new_content = []
                    for s in slices_:
                        new_content.extend(lines[s])
                    lines = new_content
                content = '\n'.join(lines)
                content = content.encode('utf8', 'ignore').decode('utf8', 'ignore')
                return content if not with_metadata else [TextContainer(id, None, content)]
            except Exception:
                # Fallback to Tika if plain read fails
                pass

        _ensure_tika_vm()
        file_ = unpack.from_file(str(path_obj))
        content = file_['content'] if 'content' in file_ else str(file_)

        if content is None:
            return None
        content = content.split('\n')

        if slices_ is not None:
            new_content = []
            for s in slices_:
                new_content.extend(content[s])
            content = new_content
        content = '\n'.join(content)
        content = content.encode('utf8', 'ignore').decode('utf8', 'ignore')
        return content if not with_metadata else [TextContainer(id, None, content)]


    def reset_eof_of_pdf_return_stream(self, pdf_stream_in: list):
        actual_line = len(pdf_stream_in)  # Predefined value in case EOF not found
        # find the line position of the EOF
        for i, x in enumerate(pdf_stream_in[::-1]):
            if b'%%EOF' in x:
                actual_line = len(pdf_stream_in)-i
                CustomUserWarning(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
                break

        # return the list up to that point
        return pdf_stream_in[:actual_line]

    def fix_pdf(self, file_path: str):
        # opens the file for reading
        path_obj = Path(file_path)
        with path_obj.open('rb') as p:
            txt = (p.readlines())

        # get the new list terminating correctly
        txtx = self.reset_eof_of_pdf_return_stream(txt)

        # write to new pdf
        new_file_path = Path(f'{file_path}_fixed.pdf')
        with new_file_path.open('wb') as f:
            f.writelines(txtx)

        return pypdf.PdfReader(str(new_file_path))

    def read_text(self, pdf_reader, page_range, argument):
        txt = []
        n_pages  = len(pdf_reader.pages)
        with_metadata = argument.kwargs.get('with_metadata', False)
        id       = Path(argument.prop.prepared_input).stem.replace(' ', '_')
        for i in range(n_pages)[slice(0, n_pages) if page_range is None else page_range]:
            page = pdf_reader.pages[i]
            extracted = page.extract_text()
            extracted = extracted.encode('utf8', 'ignore').decode('utf8', 'ignore')
            if with_metadata:
                txt.append(TextContainer(id, str(i), extracted))
            else:
                txt.append(extracted)

        return '\n'.join(txt) if not with_metadata else txt

    def forward(self, argument):
        kwargs        = argument.kwargs
        path          = argument.prop.prepared_input

        if '.pdf' in path:
            page_range = None
            if 'slice' in kwargs:
                page_range = kwargs['slice']
                if isinstance(page_range, (tuple, list)):
                    page_range = slice(*page_range)

            rsp = ''
            try:
                with Path(path).open('rb') as f:
                    # creating a pdf reader object
                    pdf_reader = pypdf.PdfReader(f)
                    rsp = self.read_text(pdf_reader, page_range, argument)
            except Exception as e:
                CustomUserWarning(f'Error reading PDF: {e} | {path}')
                if 'fix_pdf' not in kwargs or not kwargs['fix_pdf']:
                    raise e
                fixed_pdf = self.fix_pdf(str(path))
                pdf_reader_fixed = pypdf.PdfReader(fixed_pdf)
                rsp = self.read_text(pdf_reader_fixed, page_range, argument)
        else:
            try:
                rsp = self._read_slice_file(path, argument)
            except Exception as e:
                CustomUserWarning(f'Error reading empty file: {e} | {path}')
                raise e

        if rsp is None:
            CustomUserWarning(f'Error reading file - empty result: {path}', raise_with=Exception)

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "FileEngine does not support processed_input."
        path = argument.prop.path
        path = path.replace('\\', '')
        argument.prop.prepared_input = path
