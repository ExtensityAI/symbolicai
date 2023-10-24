import pygments
from pygments.lexers.python import PythonLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.c_cpp import CppLexer
from pygments.lexers.shell import BashLexer
from prompt_toolkit import print_formatted_text
from prompt_toolkit import HTML
from prompt_toolkit.formatted_text import PygmentsTokens


print = print_formatted_text


class ConsoleStyle(object):
    style_types = {
        'alert':   'ansired',
        'error':   'ansired',
        'warn':    'ansiyellow',
        'info':    'ansiblue',
        'success': 'ansigreen',
        'debug':   'ansigray',
        'custom':  'custom',
        'code': 'code',
        'default': '',
    }

    def __init__(self, style_type = '', color = ''):
        self.style_type = style_type
        self.color = color

    def __call__(self, message):
        self.print(message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def print(self, message):
        message = str(message)
        style = self.style_types.get(self.style_type, self.style_types['default'])
        # check if message is code
        if style == self.style_types['code'] or any([lang in message.lower() for lang in ['python', 'javascript', 'typescript', 'c++', 'bash', 'shell']]):
            # select lexer
            if 'python' in message.lower():
                lexer = PythonLexer()
            elif 'javascript' in message.lower() or 'typescript' in message.lower():
                lexer = JavascriptLexer()
            elif 'c++' in message.lower():
                lexer = CppLexer()
            else:
                lexer = BashLexer()

            if '```' in message:
                messages = message.split('```')
                for i, msg in enumerate(messages):
                    if i % 2 == 0:
                        print(msg)
                    else:
                        print('```', end='')
                        tokens = list(pygments.lex(msg, lexer=lexer))
                        print(PygmentsTokens(tokens))
                        print('```')

            else:
                tokens = list(pygments.lex(message, lexer=lexer))
                print(PygmentsTokens(tokens))
            return

        if style == self.style_types['default']:
            print(message)
        elif style == self.style_types['custom']:
            print(HTML(f'<{self.color}>{message}</{self.color}>'))
        else:
            print(HTML(f'<{style}>{message}</{style}>'))
