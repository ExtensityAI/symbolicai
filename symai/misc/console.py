import logging
import re

#@TODO: refactor to use rich instead of prompt_toolkit
from html import escape as escape_html
from typing import ClassVar

import pygments
from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.formatted_text import PygmentsTokens
from pygments.lexers.c_cpp import CppLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.shell import BashLexer

logger = logging.getLogger(__name__)
print = print_formatted_text # noqa


class ConsoleStyle:
    style_types: ClassVar[dict[str, str]] = {
        'alert':     'orange',
        'error':     'ansired',
        'warn':      'ansiyellow',
        'info':      'ansiblue',
        'success':   'ansigreen',
        'extensity': '#009499',
        'text':      'ansigray',
        'debug':     'gray',
        'custom':    'custom',
        'code':      'code',
        'default':   '',
    }

    def __init__(self, style_type = '', color = '', logging: bool = False):
        self.style_type = style_type
        self.color      = color
        self.logging    = logging

    def __call__(self, message):
        self.print(message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def print(self, message, escape: bool = False):
        message = str(message)
        if self.logging:
            logger.debug(message)
        # Prepare safe content for HTML printing without mutating the original
        content_for_html = escape_html(message) if escape else message
        style = self.style_types.get(self.style_type, self.style_types['default'])

        if style == self.style_types['code']:
            self._print_code_message(message)
            return
        if style == self.style_types['default']:
            print(message)
            return
        if style == self.style_types['custom']:
            self._print_html(self.color, content_for_html)
            return
        self._print_html(style, content_for_html)

    def _print_code_message(self, message: str) -> None:
        segments = re.split(r'(```)', message)
        is_code_segment = False
        for segment in segments:
            if segment == '```':
                is_code_segment = not is_code_segment
                continue
            if is_code_segment:
                self._print_code_segment(segment)
                continue
            print(segment, end='\n')

    def _print_code_segment(self, segment: str) -> None:
        lexer = self._select_lexer(segment)
        tokens = list(pygments.lex("```" + segment + "```", lexer))
        print(PygmentsTokens(tokens), end='')

    def _select_lexer(self, segment: str):
        lowered_segment = segment.lower()
        if 'python' in lowered_segment:
            return PythonLexer()
        if 'javascript' in lowered_segment or 'typescript' in lowered_segment:
            return JavascriptLexer()
        if 'c++' in lowered_segment:
            return CppLexer()
        return BashLexer()

    def _print_html(self, color: str, content: str) -> None:
        print(HTML(f'<style fg="{color}">{content}</style>'))
