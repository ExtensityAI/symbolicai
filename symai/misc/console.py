import sys

from colorama import Fore, Style


class ConsoleStyle:
    style_types = {
        'alert': Fore.RED + Style.BRIGHT,
        'error': Fore.RED + Style.BRIGHT,
        'warn': Fore.YELLOW + Style.BRIGHT,
        'info': Fore.LIGHTCYAN_EX,
        'success': Fore.GREEN + Style.BRIGHT,
        'debug': Fore.LIGHTBLACK_EX,
        'reset': Style.RESET_ALL,
    }

    def __init__(self, style_type = 'info'):
        self.style_start = self.style_types.get(style_type, '')
        self.style_end = Style.RESET_ALL if style_type in self.style_types else ''

    def __enter__(self):
        sys.stdout.write(self.style_start)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write(self.style_end)

    def print(self, message):
        print(message)  # The style will be added due to the __enter__ and __exit__ methods
