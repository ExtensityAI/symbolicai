import os
import subprocess
from typing import Iterable
from prompt_toolkit import PromptSession
from prompt_toolkit.history import History
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.keys import Keys
from prompt_toolkit import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from symai.misc.console import ConsoleStyle
from symai.misc.loader import Loader
from symai.components import Function
from symai.symbol import Symbol


SHELL_CONTEXT = """[Description]
This shell program is the command interpreter on the Linux systems, MacOS and Windows PowerShell.
It is a program that interacts with the users in the terminal emulation window.
Shell commands are instructions that instruct the system to do some action.

[Program Instructions]
If user requests commands, you will only process user queries and return valid Linux/MacOS shell or Windows PowerShell commands and no other text.
If no further specified, use as default the Linux/MacOS shell commands.
If additional instructions are provided the follow the user query to produce the requested output.
"""


# Create custom keybindings
bindings = KeyBindings()


def get_conda_env():
    return os.environ.get('CONDA_DEFAULT_ENV')


# Define what happens when 'Ctrl + C' is pressed
@bindings.add(Keys.ControlC)
def _(event):
    event.current_buffer.cancel_completion()
    event.app.current_buffer.reset()


@bindings.add(Keys.Tab)
def _(event):
    event.current_buffer.cancel_completion()
    event.app.current_buffer.reset()
    print('penis')


class FileHistory(History):
    '''
    :class:`.History` class that stores all strings in a file.
    '''

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__()

    def load_history_strings(self) -> Iterable[str]:
        lines: list[str] = []

        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                lines = f.readlines()
                # Remove comments and empty lines.
                lines = [line for line in lines if line.strip() and not line.startswith("#")]
                # Remove whitespace at the end and \n.
                lines = [line.rstrip() for line in lines]
                # only keep unique lines
                lines = list(dict.fromkeys(lines))

        # Reverse the order, because newest items have to go first.
        return reversed(lines)

    def store_string(self, string: str) -> None:
        # Save to file.
        with open(self.filename, "ab") as f:

            def write(t: str) -> None:
                f.write(t.encode("utf-8"))

            for line in string.split("\n"):
                write("%s\n" % line)


# Defining commands history
def load_history(home_path=os.path.expanduser('~'), history_file=f'.bash_history'):
    history_file_path = os.path.join(home_path, history_file)
    history = FileHistory(history_file_path)
    return history, list(history.load_history_strings())


# Function to check if current directory is a git directory
def get_git_branch():
    if os.path.exists('.git') or os.system('git rev-parse > /dev/null 2>&1') == 0:
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    return None


# query language model
def query_language_model(query: str, *args, **kwargs):
    with Loader(desc="Inference ...", end=""):
        func = Function(SHELL_CONTEXT)
        msg = func(query, *args, **kwargs)

    with ConsoleStyle('info'):
        print(msg)


# run shell command
def run_shell_command(cmd: str):
    # Execute the command
    res = subprocess.run(cmd, shell=True)

    # If command not found, then try to query language model
    if res.returncode != 0:
        msg = Symbol(cmd) @ str(res)
        if not ('command not found' in str(res)):
            query_language_model(msg)


# Function to listen for user input and execute commands
def listen(session: PromptSession, word_comp: WordCompleter):
    with patch_stdout():
        while True:
            try:
                git_branch = get_git_branch()
                conda_env = get_conda_env()
                # get directory from the shell
                cur_working_dir = os.getcwd()
                if cur_working_dir.startswith('/'):
                    cur_working_dir = cur_working_dir.replace(os.path.expanduser('~'), '~')
                paths = cur_working_dir.split('/')
                prev_paths = '/'.join(paths[:-1])
                last_path = paths[-1]

                # Format the prompt
                if len(paths) > 1:
                    cur_working_dir = f'{prev_paths}/<b>{last_path}</b>'
                else:
                    cur_working_dir = f'<b>{last_path}</b>'

                if git_branch:
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue><ansiwhite> on git:[</ansiwhite><ansigreen>{git_branch}</ansigreen><ansiwhite>]</ansiwhite> <ansiwhite>conda:[</ansiwhite><ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansired><b>symsh:</b> ❯</ansired> ")
                else:
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue> <ansiwhite>conda:[</ansiwhite><ansigray>{conda_env}</ansigray><ansiwhite>]</ansiwhite> <ansired><b>symsh:</b> ❯</ansired> ")

                # Read user input
                cmd = session.prompt(prompt)
                if cmd.strip() == '':
                    continue

                if cmd == 'quit' or cmd == 'exit':
                    os._exit(0)
                elif cmd.startswith('"') or cmd.startswith("'"):
                    query_language_model(cmd)
                elif cmd.startswith('cd'):
                    try:
                        # replace ~ with home directory
                        cmd = cmd.replace('~', os.path.expanduser('~'))
                        # Change directory
                        os.chdir(cmd.split(' ')[1])
                    except FileNotFoundError as e:
                        print(e)
                elif cmd.startswith('ll'):
                    run_shell_command('ls -l')
                else:
                    run_shell_command(cmd)

            except KeyboardInterrupt:
                pass


def run():
    # Load history
    history, history_strings = load_history()

    # Defining the auto-completion words (according to history)
    word_comp = WordCompleter(history_strings, ignore_case=True, sentence=True)

    style = Style.from_dict({
        "completion-menu.completion": "bg:#a33a33 #ffffff",
        "completion-menu.completion.current": "bg:#aaccaa #000000",
        "scrollbar.background": "bg:#222222",
        "scrollbar.button": "bg:#776677",
    })

    # Session for the auto-completion
    session = PromptSession(history=history,
                            completer=word_comp,
                            complete_style=CompleteStyle.COLUMN,
                            style=style)

    listen(session, word_comp)


if __name__ == '__main__':
    run()
