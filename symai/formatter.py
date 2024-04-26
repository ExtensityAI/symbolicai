import json
import re

from beartype import beartype
from beartype.typing import Any, Dict, List
from tqdm import tqdm

from . import core_ext
from .symbol import Expression, Symbol


class ParagraphFormatter(Expression):
    def __init__(self, value=None, **kwargs):
        super().__init__(value, **kwargs)
        self.NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters
        self._has_file_start = False

    def split_files(self, input_text=""):
        input_ = input_text.strip()
        if input_.startswith('# ----[FILE_START]') and '# ----[FILE_END]' in input_:
            self._has_file_start = True
            # split text file-wise and create a map of file names and their contents
            files = {}
            split_text = input_.split('# ----[FILE_START]')
            for i, file in enumerate(split_text):
                if not file.strip():
                    continue
                _, content_file = file.split('[FILE_CONTENT]:')
                content, file_name = content_file.split('# ----[FILE_END]')
                files[file_name.strip()] = content.strip()
        else:
            files = {"": input_}
        return files

    def _add_header_footer(self, paragraph, file_name, part=1, total_parts=1):
        if file_name and self._has_file_start:
            header = f"# ----[FILE_START]<PART{part}/{total_parts}>{file_name}[FILE_CONTENT]:\n"
            footer = f"\n# ----[FILE_END]{file_name}\n"
            if '[FILE_CONTENT]:' in paragraph: # TODO: remove this if statement after fixing the bug
                paragraph = paragraph.split('[FILE_CONTENT]:')[-1].strip()
            paragraph = header + paragraph + footer
        return paragraph

    def _get_file_name(self, paragraph):
        if not self._has_file_start:
            return ""
        return paragraph.split("# ----[FILE_END]")[-1].strip()

    def _get_part(self, paragraph):
        if not self._has_file_start:
            return 1
        return int(paragraph.split("# ----[FILE_START]<")[-1].split("/")[0].split(">")[0])

    def _get_total_parts(self, paragraph):
        if not self._has_file_start:
            return 1
        return int(paragraph.split("# ----[FILE_START]<")[-1].split("/")[1].split(">")[0])

    def split_paragraphs(self, input_text: Dict[str, str]):
        paragraphs = []

        for file_name, file_content in input_text.items():
            input_ = file_content.strip()
            split_text = self.NEWLINES_RE.split(input_)

            par = [self._add_header_footer(p, file_name, part=i+1, total_parts=len(split_text)) + "\n" for i, p in enumerate(split_text) if p.strip()]
            # p + "\n" ensures that all lines in the paragraph end with a newline
            # p.strip() == True if paragraph has other characters than whitespace

            paragraphs.extend(par)

        return paragraphs

    def split_huge_paragraphs(self, input_text: List[str], max_length=500):
        paragraphs = []
        for text in input_text:
            words = text.split()
            if len(words) > max_length:
                # get file name
                file_name = self._get_file_name(text)
                # n splits
                total_parts = (len(words) // max_length + 1) * self._get_total_parts(text)
                for p, i in enumerate(range(0, len(words), max_length)):
                    paragraph = ' '.join(words[i:i + max_length])
                    paragraphs.append(self._add_header_footer(paragraph, file_name, part=p+1, total_parts=total_parts) + "\n")
            else:
                paragraphs.append(text)
        return paragraphs

    @core_ext.bind(engine='embedding', property='max_tokens')
    def _max_tokens(self): pass

    def split_max_tokens_exceeded(self, input_text: List[str], token_ratio=0.5):
        paragraphs = []
        max_ctxt_tokens = int(self._max_tokens() * token_ratio)
        for text in input_text:
            len_ = len(self.tokenizer().encode(text, disallowed_special=()))
            if len_ > max_ctxt_tokens:
                # get file name
                file_name = self._get_file_name(text)
                # split into chunks of max_ctxt_tokens
                splits_ = len_ // max_ctxt_tokens
                text_len_ = len(str(text)) // splits_
                total_parts = (text_len_ + 1) * self._get_total_parts(text)
                for i in range(splits_):
                    paragraph = text[i * text_len_:(i + 1) * text_len_]
                    paragraphs.append(self._add_header_footer(paragraph, file_name, part=i+1, total_parts=total_parts) + "\n")
            else:
                paragraphs.append(text)
        return paragraphs

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        # split text paragraph-wise and index each paragraph separately
        self.elements = self.split_files(sym.value)
        self.elements = self.split_paragraphs(self.elements)
        self.elements = self.split_huge_paragraphs(self.elements)
        self.elements = self.split_max_tokens_exceeded(self.elements)
        return self._to_symbol(self.elements)


class SentenceFormatter(Expression):
    def __init__(self, value=None, **kwargs):
        super().__init__(value, **kwargs)
        self.SENTENCES_RE = re.compile(r"[.!?]\n*|[\n]{1,}")  # Sentence ending characters followed by newlines

    def split_sentences(self, input_text=""):
        input_ = input_text.strip()
        split_text = self.SENTENCES_RE.split(input_)  # regex splitting

        sentences = [s.strip() + ".\n" for s in split_text if s.strip()]
        # s.strip() + ".\n" ensures that all lines in the sentence end with a period and newline
        # s.strip() == True if sentence has other characters than whitespace

        return sentences

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        # split text sentence-wise and index each sentence separately
        self.elements = self.split_sentences(sym.value)
        return self._to_symbol(self.elements)


class TextContainerFormatter(Expression):
    def __init__(
            self,
            value: Any = None,
            key: str ="text",
            text_split: int = 4,
            **kwargs
        ):
        super().__init__(value, **kwargs)
        self.key = key
        self.text_split = text_split

    @beartype
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        if isinstance(sym.value, list):
            containers = [container for pdf in sym.value for container in pdf]
        chunks = [text for container in tqdm(containers) for text in self._chunk(container)]
        return self._to_symbol(chunks)

    def _chunk(self, container: 'TextContainer') -> List[str]:
        text = container.text
        step = len(text) // self.text_split
        splits = []
        i = c = 0
        while c < self.text_split:
            if c == self.text_split - 1:
                # Unify the last chunk with the previous one if necessary
                splits.append(self._as_str(text[i:], container))
                break
            splits.append(self._as_str(text[i:i+step], container))
            i += step
            c += 1
        return splits

    def _as_str(self, text: str, container: 'TextContainer') -> str:
        return (
            '---\n'
            f"id: {container.id}\n"
            f"page: {container.page}\n"
            '---\n'
            f"{text}"
        )

