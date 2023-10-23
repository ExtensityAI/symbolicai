import os
import subprocess
import glob
from typing import Iterable
from prompt_toolkit.completion import Completer, Completion
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


current_command = None
current_language_model_result = None


def supports_ansi_escape():
    try:
        os.get_terminal_size(0)
        return True
    except OSError:
        return False


class PathCompleter(Completer):
    def get_completions(self, document, complete_event):
        complete_word = document.get_word_before_cursor(WORD=True)
        if complete_word.startswith('~/'):
            complete_word = complete_word.replace('~/', os.path.expanduser('~'))

        files = glob.glob(complete_word + '*')

        for file in files:
            # split the command into words by space (ignore escaped spaces)
            command_words = document.text.split(' ')
            if len(command_words) > 1:
                # Calculate start position of the completion
                start_position = len(document.text) - len(' '.join(command_words[:-1])) - 1
                start_position = max(0, start_position)
            else:
                start_position = len(document.text)
            # if there is a space in the file name, then escape it
            if ' ' in file:
                file = file.replace(' ', '\\ ')
            if (document.text.startswith('cd') or document.text.startswith('mkdir')) and os.path.isfile(file):
                continue

            yield Completion(file, start_position=-start_position,
                             style='class:path-completion',
                             selected_style='class:path-completion-selected')


class HistoryCompleter(WordCompleter):
    def get_completions(self, document, complete_event):
        completions = super().get_completions(document, complete_event)
        for completion in completions:
            completion.style = 'class:history-completion'
            completion.selected_style = 'class:history-completion-selected'
            yield completion


class MergedCompleter(Completer):
    def __init__(self, *completers):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for completer in self.completers:
            for completion in completer.get_completions(document, complete_event):
                yield completion


# Create custom keybindings
bindings = KeyBindings()


def get_conda_env():
    return os.environ.get('CONDA_DEFAULT_ENV')


# bind to 'Ctrl' + 'Space'
@bindings.add(Keys.ControlSpace)
def _(event):
    current_user_input = event.current_buffer.document.text
    query_language_model(current_user_input)


@bindings.add(Keys.PageUp)
def _(event):
    # Moving up for 5 lines
    for i in range(5):
        event.current_buffer.auto_up()


@bindings.add(Keys.PageDown)
def _(event):
    # Moving down for 5 lines
    for i in range(5):
        event.current_buffer.auto_down()


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
    global current_language_model_result
    func = Function(SHELL_CONTEXT)
    with Loader(desc="Inference ...", end=""):
        msg = func(query, *args, **kwargs)

    with ConsoleStyle('info'):
        print(msg)

    current_language_model_result = msg


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
    global current_command
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
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue><ansiwhite> on git:[</ansiwhite><ansigreen>{git_branch}</ansigreen><ansiwhite>]</ansiwhite> <ansiwhite>conda:[</ansiwhite><ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ❯</ansicyan> ")
                else:
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue> <ansiwhite>conda:[</ansiwhite><ansigray>{conda_env}</ansigray><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ❯</ansicyan> ")

                # Read user input
                cmd = session.prompt(prompt)
                if cmd.strip() == '':
                    continue
                current_command = cmd

                # Append the command to the word completer list
                word_comp.words.append(cmd)

                if cmd == 'quit' or cmd == 'exit':
                    os._exit(0)
                elif cmd.startswith('"') or cmd.startswith("'"):
                    query_language_model(cmd)
                elif cmd.startswith('cd') or cmd.startswith('mkdir'):
                    try:
                        # replace ~ with home directory
                        cmd = cmd.replace('~', os.path.expanduser('~'))
                        # Change directory
                        os.chdir(cmd.split(' ')[1])
                    except FileNotFoundError as e:
                        print(e)
                elif os.path.isdir(cmd):
                    try:
                        # replace ~ with home directory
                        cmd = cmd.replace('~', os.path.expanduser('~'))
                        # Change directory
                        os.chdir(cmd)
                    except FileNotFoundError as e:
                        print(e)
                elif cmd.startswith('ll'):
                    run_shell_command('ls -l')
                else:
                    run_shell_command(cmd)

            except KeyboardInterrupt:
                pass


def run():
    if not supports_ansi_escape():
        print("Unfortunately, your terminal does not support ANSI escape sequences.")

    # Load history
    history, history_strings = load_history()

    # Create your specific completers
    word_comp = HistoryCompleter(history_strings, ignore_case=True, sentence=True)
    custom_completer = PathCompleter()

    # Merge completers
    merged_completer = MergedCompleter(custom_completer, word_comp)

    style = Style.from_dict({
        "completion-menu.completion.current": "bg:#323232 #000080",  # Change to your preference
        "completion-menu.completion": "bg:#800080 #000080",
        "scrollbar.background": "bg:#222222",
        "scrollbar.button": "bg:#776677",
        "history-completion": "bg:#323232 #efefef",
        "path-completion": "bg:#800080 #efefef",
        "history-completion-selected": "bg:#efefef #000000",
        "path-completion-selected": "bg:#efefef #800080",
    })

    # Session for the auto-completion
    session = PromptSession(history=history,
                            completer=merged_completer,
                            complete_style=CompleteStyle.MULTI_COLUMN,
                            style=style,
                            key_bindings=bindings)

    listen(session, word_comp)


if __name__ == '__main__':
    run()
