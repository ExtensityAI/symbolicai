import re
from itertools import takewhile
from typing import List, Dict

from . import core
from .symbol import Expression, Symbol


class ParagraphFormatter(Expression):
    def __init__(self, value=None):
        super().__init__(value)
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

    @core.bind(engine='embedding', property='max_tokens')
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
    def __init__(self, value=None):
        super().__init__(value)
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


class WhisperTimestampsFormatter(Expression):
    def __init__(self):
        super().__init__()

    def forward(self, response: List[str]) -> str:
        result = []
        for i, interval in enumerate(response):
            interval = self._filter_empty_string(interval)
            prev_end = 0.0
            prev_start = 0.0
            for head, tail in zip(interval[::2], interval[1::2]):
                start = self._get_timestamp(head)
                end = self._get_timestamp(tail)
                if start >= prev_end:
                    start = prev_end
                    prev_end = end
                    prev_start = start
                    result.append(f"{self._format_to_hours(start + (i*30))} {self._get_sentence(head)}")
                    continue
                if start < prev_start:
                    continue
                delta = end - start
                if start + prev_end > 30:
                    start = prev_end
                else:
                    start += prev_end
                if start + delta > 30:
                    end = 30
                else:
                    end = start + delta
                prev_end = end
                result.append(f"{self._format_to_hours(start + (i*30))} {self._get_sentence(head)}")
        return "\n".join(result)

    def _filter_empty_string(self, s: str) -> List[str]:
        return list(filter(lambda x: x, s.split("<|")))

    def _get_timestamp(self, s: str) -> float:
        return float("".join(list(takewhile(lambda x: x != "|", s))))

    def _get_sentence(self, s: str) -> str:
        return s.split("|>")[-1]

    def _format_to_hours(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds))
        return formatted_time

