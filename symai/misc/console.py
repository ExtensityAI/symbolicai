import re
import pygments
import logging

from html import escape as escape_html
from pygments.lexers.python import PythonLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.c_cpp import CppLexer
from pygments.lexers.shell import BashLexer
from prompt_toolkit import print_formatted_text
from prompt_toolkit import HTML
from prompt_toolkit.formatted_text import PygmentsTokens


logger = logging.getLogger(__name__)
print = print_formatted_text


class ConsoleStyle(object):
    style_types = {
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
        if escape:
            message = escape_html(message)
        style = self.style_types.get(self.style_type, self.style_types['default'])

        if style == self.style_types['code']:
            # Split the message using ``` as delimiter
            segments = re.split(r'(```)', message)
            is_code = False
            for segment in segments:
                if segment == '```':
                    is_code = not is_code
                    continue
                if is_code:
                    # Determine lexer
                    if 'python' in segment.lower():
                        lexer = PythonLexer()
                    elif 'javascript' in segment.lower() or 'typescript' in segment.lower():
                        lexer = JavascriptLexer()
                    elif 'c++' in segment.lower():
                        lexer = CppLexer()
                    else:
                        lexer = BashLexer()
                    # Print highlighted code
                    tokens = list(pygments.lex("```" + segment + "```", lexer))
                    print(PygmentsTokens(tokens), end='')
                else:
                    # Print the segment normally
                    print(segment, end='\n')
        elif style == self.style_types['default']:
            print(message)
        elif style == self.style_types['custom']:
            print(HTML(f'<style fg="{self.color}">{message}</style>'))
        else:
            print(HTML(f'<style fg="{style}">{message}</style>'))

