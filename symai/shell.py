import argparse
import os

from .components import Lambda, Try
from .core import few_shot
from .misc.console import ConsoleStyle
from .misc.loader import Loader
from .post_processors import StripPostProcessor
from .pre_processors import PreProcessor
from .symbol import Expression
from .shellsv import run as shellsv_run
from . import SYMAI_VERSION


SHELL_CONTEXT = """[Description]
This shell program is the command interpreter on the Linux systems, MacOS and Windows PowerShell.
It is a program that interacts with the users in the terminal emulation window.
Shell commands are instructions that instruct the system to do some action.

[Program Instructions]
You will only process user queries and return valid Linux/MacOS shell or Windows PowerShell commands and no other text.
EOF is the end of file character and marks the end of the output command.
If you do not understand the user query, then return an sorry message and ask the user to rephrase the query.
If no further specified, use as default the Linux/MacOS shell commands.
Additional instructions are provided in examples or via dynamic context in the prompt.

[Examples]
Some of the commonly used shell commands are:
// List the contents of the current directory in long format.
$> ls -l EOF
// This command concatenates and prints the contents of the file. If there is no file, then it reads the standard input
$> cat file EOF
// This command is used to create a new directory called helloworld.
$> mkdir helloworld EOF
// command to copy the /path/to/file1 to /second/path/to/other/file2.
$> cp /path/to/file1 /second/path/to/other/file2 EOF
// Linux env variable OPENAI_API_KEY with value "<openai_api_key>".
$> export OPENAI_API_KEY=<openai_api_key> EOF
// Set Windows env variable OPENAI_API_KEY with value "<openai_api_key>".
$> $Env:OPENAI_API_KEY="<openai_api_key>" EOF
// Create a Directory in PowerShell
$> New-Item -ItemType Directory -Path <path> EOF

[Last Example]
--------------
"""

class ShellPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '// {}\n$>'.format(str(argument.prop.instance))


class Shell(Expression):
    def forward(self, **kwargs) -> str:
        @few_shot(prompt="Convert a user query to a shell command:\n",
                     examples=[],
                     pre_processors=[ShellPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=['EOF'], **kwargs)
        def _func(_) -> str:
            return "Sorry, something went wrong. Please check if your backend is available and try again or report an issue to the devs. :("

        return Shell(_func(self))

    @property
    def static_context(self) -> str:
        return SHELL_CONTEXT


def process_query(args) -> None:
    if args.version:
        with ConsoleStyle('extensity') as console:
            console.print(f"ExtensityAI: Symbolic Shell v{SYMAI_VERSION}")
        return

    query = args.query
    if query is None or len(query) == 0:
        with ConsoleStyle('extensity') as console:
            console.print(f"Starting ExtensityAI: Symbolic Shell v{SYMAI_VERSION}")
        shellsv_run(auto_query_on_error=args.auto,
                    conversation_style=args.style if args.style is not None and args.style != '' else None,
                    verbose=args.verbose)
        return

    shell = Shell(query)

    with Loader(desc="Inference ...", end=""):
        msg = shell()

    if args.edit:
        msg = msg.modify(args.edit)
    if args.convert:
        msg = msg.convert(args.convert)
    if args.add:
        msg = msg << args.add
    if args.delete:
        msg = msg - args.delete
    if args.exec:
        msg = str(msg).strip().replace('EOF', '')
        os.system(msg)
    if args.more:
        cmd = msg.extract('the main command used')
        cmd_help = cmd << 'get manual for the command'
        expr = Try(expr=Lambda(lambda kwargs: os.system(str(kwargs['args'][0]))))
        with Loader(desc="Inference ...", end=""):
            expr(cmd_help)

    with ConsoleStyle('code') as console:
        msg = str(msg).strip().replace('EOF', '')
        console.print(msg)


def run() -> None:
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description='Welcome to the Symbolic<AI/> Shell support tool!')
    parser.add_argument('query', type=str, nargs='?', help='The prompt for the shell query.')
    parser.add_argument('--add', dest='add', default="", required=False, type=str,
                        help='integrate the added text to the query.')
    parser.add_argument('--convert', dest='convert', default="", required=False, type=str,
                        help='convert a command to another shell. (e.g. --convert=windows or --convert=linux)')
    parser.add_argument('--edit', dest='edit', default="", required=False, type=str,
                        help='edit the added text to the query.')
    parser.add_argument('--del', dest='delete', default="", required=False, type=str,
                        help='remove the added text to the query.')
    parser.add_argument('--more', dest='more', default=False, required=False, action=argparse.BooleanOptionalAction,
                        help='add more context to the generated command.')
    parser.add_argument('--exec', dest='exec', default=False, required=False, action=argparse.BooleanOptionalAction,
                        help='execute command after creation (ATTENTION: Executing a generated command without verification may be risky!).')
    parser.add_argument('--auto', dest='auto', default=False, required=False, action=argparse.BooleanOptionalAction,
                        help='query the the LLM if a command resulted unsuccessfully.')
    parser.add_argument('--style', dest='style', default=None, required=False, type=str,
                        help='add speech style to the shell.')
    parser.add_argument('--version', dest='version', default=False, required=False, action=argparse.BooleanOptionalAction,
                        help='show the version of the shell.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose errors.')

    args = parser.parse_args()
    process_query(args)
