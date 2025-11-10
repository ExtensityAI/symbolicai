import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pypdf
import tika
from tika import unpack

from ....utils import UserMessage
from ...base import Engine

# Initialize Tika lazily to avoid spawning JVMs prematurely for all workers
_TIKA_STATE = {"initialized": False}

def _ensure_tika_vm():
    if not _TIKA_STATE["initialized"]:
        with contextlib.suppress(Exception):
            tika.initVM()
        logging.getLogger('tika').setLevel(logging.CRITICAL)
        _TIKA_STATE["initialized"] = True


def _int_or_none(value):
    return int(value) if value != '' else None


def _parse_slice_token(token):
    if ':' not in token:
        return int(token)
    parts = token.split(':')
    if len(parts) == 2:
        start, end = parts
        return slice(_int_or_none(start), _int_or_none(end), None)
    if len(parts) == 3:
        start, end, step = parts
        return slice(_int_or_none(start), _int_or_none(end), _int_or_none(step))
    return None


def _parse_slice_spec(file_path):
    if '[' not in file_path or ']' not in file_path:
        return file_path, None
    path_part, remainder = file_path.split('[', 1)
    slice_section = remainder.split(']', 1)[0]
    slices = []
    for token in slice_section.split(','):
        if token == '':
            continue
        parsed = _parse_slice_token(token)
        if parsed is not None:
            slices.append(parsed)
    return path_part, slices or None


def _apply_slices(lines, slices_):
    if slices_ is None:
        return lines
    new_content = []
    for slice_item in slices_:
        new_content.extend(lines[slice_item])
    return new_content


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
        file_id       = Path(argument.prop.prepared_input).stem.replace(' ', '_')
        if file_path is None or file_path.strip() == '':
            return None

        # check if file slice is used
        file_path, slices_ = _parse_slice_spec(file_path)

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
                lines = _apply_slices(lines, slices_)
                content = '\n'.join(lines)
                content = content.encode('utf8', 'ignore').decode('utf8', 'ignore')
                return content if not with_metadata else [TextContainer(file_id, None, content)]
            except Exception:
                # Fallback to Tika if plain read fails
                pass

        _ensure_tika_vm()
        file_ = unpack.from_file(str(path_obj))
        content = file_['content'] if 'content' in file_ else str(file_)

        if content is None:
            return None
        content = content.split('\n')

        content = _apply_slices(content, slices_)
        content = '\n'.join(content)
        content = content.encode('utf8', 'ignore').decode('utf8', 'ignore')
        return content if not with_metadata else [TextContainer(file_id, None, content)]


    def reset_eof_of_pdf_return_stream(self, pdf_stream_in: list):
        actual_line = len(pdf_stream_in)  # Predefined value in case EOF not found
        # find the line position of the EOF
        for i, x in enumerate(pdf_stream_in[::-1]):
            if b'%%EOF' in x:
                actual_line = len(pdf_stream_in)-i
                UserMessage(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
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
        file_id       = Path(argument.prop.prepared_input).stem.replace(' ', '_')
        for i in range(n_pages)[slice(0, n_pages) if page_range is None else page_range]:
            page = pdf_reader.pages[i]
            extracted = page.extract_text()
            extracted = extracted.encode('utf8', 'ignore').decode('utf8', 'ignore')
            if with_metadata:
                txt.append(TextContainer(file_id, str(i), extracted))
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
                UserMessage(f'Error reading PDF: {e} | {path}')
                if 'fix_pdf' not in kwargs or not kwargs['fix_pdf']:
                    raise e
                fixed_pdf = self.fix_pdf(str(path))
                pdf_reader_fixed = pypdf.PdfReader(fixed_pdf)
                rsp = self.read_text(pdf_reader_fixed, page_range, argument)
        else:
            try:
                rsp = self._read_slice_file(path, argument)
            except Exception as e:
                UserMessage(f'Error reading empty file: {e} | {path}')
                raise e

        if rsp is None:
            UserMessage(f'Error reading file - empty result: {path}', raise_with=Exception)

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "FileEngine does not support processed_input."
        path = argument.prop.path
        path = path.replace('\\', '')
        argument.prop.prepared_input = path
