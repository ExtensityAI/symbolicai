import os

from prompt_toolkit import print_formatted_text

from ..misc.console import ConsoleStyle


def show_splash_screen(print: callable = print_formatted_text):
    print('\n\n')
    print('- '*42)
    print('=='*41 + '=')
    print(r'''
      ____|        |                      _)  |              \    _ _|
      __|  \ \  /  __|   _ \  __ \    __|  |  __|  |   |    _ \     |
      |     `  <   |     __/  |   | \__ \  |  |    |   |   ___ \    |
     _____| _/\_\ \__| \___| _|  _| ____/ _| \__| \__, | _/    _\ ___|
                                                  ____/
    ''', escape=True)
    print('- '*42)


def show_info_message(print: callable = print_formatted_text):
    print('Welcome to SymbolicAI!' + '\n')
    print('SymbolicAI is an open-source Python project for building AI-powered applications\nand assistants.')
    print('We utilize the power of large language models and the latest research in AI.' + '\n')
    print('SymbolicAI is backed by ExtensityAI. We are committed to open research,\nthe democratization of AI tools and much more ...' + '\n')

    print('... and we also like peanut butter and jelly sandwiches, and cookies.' + '\n\n')
    print('If you like what we are doing please help us achieve our mission!')
    print('More information is available at https://www.extensity.ai' + '\n')


def show_separator(print: callable = print_formatted_text):
    print('- '*42 + '\n')


def show_intro_menu():
    if os.environ.get('SYMAI_WARNINGS', '1') == '1':
        with ConsoleStyle('extensity') as console:
            show_splash_screen(print=console.print)
        with ConsoleStyle('text') as console:
            show_info_message(print=console.print)
        with ConsoleStyle('extensity') as console:
            show_separator(print=console.print)

if __name__ == '__main__':
    show_intro_menu()
