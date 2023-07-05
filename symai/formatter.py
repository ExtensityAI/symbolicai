import re
from typing import List

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

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        # split text paragraph-wise and index each paragraph separately
        self.elements = self.split_paragraphs(sym.value)
        self.elements = self.split_huge_paragraphs(self.elements)
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
