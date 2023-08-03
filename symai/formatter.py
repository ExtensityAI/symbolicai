import re
from itertools import takewhile
from typing import List

from . import core
from .symbol import Expression, Symbol


class ParagraphFormatter(Expression):
    def __init__(self, value=None):
        super().__init__(value)
        self.NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

    def split_paragraphs(self, input_text=""):
        input_ = input_text.strip()
        split_text = self.NEWLINES_RE.split(input_)  # regex splitting

        paragraphs = [p + "\n" for p in split_text if p.strip()]
        # p + "\n" ensures that all lines in the paragraph end with a newline
        # p.strip() == True if paragraph has other characters than whitespace

        return paragraphs

    def split_huge_paragraphs(self, input_text: List[str], max_length=300):
        paragraphs = []
        for text in input_text:
            words = text.split()
            if len(words) > max_length:
                for i in range(0, len(words), max_length):
                    paragraph = ' '.join(words[i:i + max_length])
                    paragraphs.append(paragraph + "\n")
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
                # split into chunks of max_ctxt_tokens
                splits_ = len_ // max_ctxt_tokens
                text_len_ = len(str(text)) // splits_
                for i in range(splits_):
                    paragraph = text[i * text_len_:(i + 1) * text_len_]
                    paragraphs.append(paragraph + "\n")
            else:
                paragraphs.append(text)
        return paragraphs

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        # split text paragraph-wise and index each paragraph separately
        self.elements = self.split_paragraphs(sym.value)
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

