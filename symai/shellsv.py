import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

#@TODO: refactor to use rich instead of prompt_toolkit
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import CompleteStyle, ProgressBar
from prompt_toolkit.styles import Style

from .backend.settings import HOME_PATH, SYMSH_CONFIG
from .components import FileReader, Function, Indexer
from .extended import (
    ArxivPdfParser,
    Conversation,
    DocumentRetriever,
    FileMerger,
    RepositoryCloner,
    RetrievalAugmentedConversation,
)
from .imports import Import
from .interfaces import Interface
from .menu.screen import show_intro_menu
from .misc.console import ConsoleStyle
from .misc.loader import Loader
from .symbol import Symbol
from .utils import UserMessage

logging.getLogger("prompt_toolkit").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("subprocess").setLevel(logging.ERROR)

# load json config from home directory root
home_path = HOME_PATH
config_path = home_path / 'symsh.config.json'
# migrate config from old path
if 'colors' not in SYMSH_CONFIG:
    __new_config__ = {"colors": SYMSH_CONFIG}
    # add command in config
    SYMSH_CONFIG = __new_config__
    # save config
    with config_path.open('w') as f:
        json.dump(__new_config__, f, indent=4)

# make sure map-nt-cmd is in config
if 'map-nt-cmd' not in SYMSH_CONFIG:
    # add command in config
    SYMSH_CONFIG['map-nt-cmd'] = True
    # save config
    with config_path.open('w') as f:
        json.dump(SYMSH_CONFIG, f, indent=4)

print = print_formatted_text # noqa
map_nt_cmd_enabled = SYMSH_CONFIG['map-nt-cmd']

_shell_state = SimpleNamespace(
    function_type=Function,
    conversation_type=Conversation,
    retrieval_conversation_type=RetrievalAugmentedConversation,
    use_styles=False,
    stateful_conversation=None,
    previous_kwargs=None,
    previous_prefix=None,
    exec_prefix="default",
)

SHELL_CONTEXT = """[Description]
This shell program is the command interpreter on the Linux systems, MacOS and Windows PowerShell.
It is a program that interacts with the users in the terminal emulation window.
Shell commands are instructions that instruct the system to do some action.

[Program Instructions]
If the user requests commands, you will only process user queries and return valid Linux/MacOS shell or Windows PowerShell commands and no other text.
If no further specified, use as default the Linux/MacOS shell commands.
If additional instructions are provided the follow the user query to produce the requested output.
A well related and helpful answer with suggested improvements is preferred over "I don't know" or "I don't understand" answers or stating the obvious.
"""

def supports_ansi_escape():
    try:
        os.get_terminal_size(0)
        return True
    except OSError:
        return False

class PathCompleter(Completer):
    def get_completions(self, document, _complete_event):
        complete_word = document.get_word_before_cursor(WORD=True)
        sep = os.path.sep
        if complete_word.startswith(f'~{sep}'):
            complete_word = FileReader.expand_user_path(complete_word)

        # list all files and directories in current directory
        complete_path = Path(complete_word)
        if complete_word.endswith(sep):
            parent = complete_path
            pattern = '*'
        else:
            baseline = Path()
            parent = complete_path.parent if complete_path.parent != baseline else baseline
            pattern = f"{complete_path.name}*" if complete_path.name else '*'
        files = [str(path) for path in parent.glob(pattern)]
        if len(files) == 0:
            return None

        dirs_ = []
        files_ = []

        for file in files:
            path_obj = Path(file)
            # split the command into words by space (ignore escaped spaces)
            command_words = document.text.split(' ')
            if len(command_words) > 1:
                # Calculate start position of the completion
                start_position = len(document.text) - len(' '.join(command_words[:-1])) - 1
                start_position = max(0, start_position)
            else:
                start_position = len(document.text)
            # if there is a space in the file name, then escape it
            display_name = file.replace(' ', '\\ ') if ' ' in file else file
            if (document.text.startswith('cd') or document.text.startswith('mkdir')) and path_obj.is_file():
                continue
            if path_obj.is_dir():
                dirs_.append(display_name)
            else:
                files_.append(display_name)

        for d in dirs_:
            # if starts with home directory, then replace it with ~
            directory_completion = FileReader.expand_user_path(d)
            yield Completion(directory_completion, start_position=-start_position,
                                 style='class:path-completion',
                                 selected_style='class:path-completion-selected')

        for f in files_:
            # if starts with home directory, then replace it with ~
            file_completion = FileReader.expand_user_path(f)
            yield Completion(file_completion, start_position=-start_position,
                                 style='class:file-completion',
                                 selected_style='class:file-completion-selected')

class HistoryCompleter(WordCompleter):
    def get_completions(self, document, complete_event):
        completions = super().get_completions(document, complete_event)
        for completion in completions:
            completion.style = 'class:history-completion'
            completion.selected_style = 'class:history-completion-selected'
            yield completion

class MergedCompleter(Completer):
    def __init__(self, path_completer, history_completer):
        self.path_completer = path_completer
        self.history_completer = history_completer

    def get_completions(self, document, complete_event):
        text = document.text

        if text.startswith('cd ') or\
            text.startswith('ls ') or\
            text.startswith('touch ') or\
            text.startswith('cat ') or\
            text.startswith('mkdir ') or\
            text.startswith('open ') or\
            text.startswith('rm ') or\
            text.startswith('git ') or\
            text.startswith('vi ') or\
            text.startswith('nano ') or\
            text.startswith('*') or\
            text.startswith(r'.\\') or\
            text.startswith(r'~\\') or\
            text.startswith(r'\\') or\
            text.startswith('.\\') or\
            text.startswith('~\\') or\
            text.startswith('\\') or\
            text.startswith('./') or\
            text.startswith('~/') or\
            text.startswith('/'):
            yield from self.path_completer.get_completions(document, complete_event)
            yield from self.history_completer.get_completions(document, complete_event)
        else:
            yield from self.history_completer.get_completions(document, complete_event)
            yield from self.path_completer.get_completions(document, complete_event)

# Create custom keybindings
bindings = KeyBindings()
# Get a copy of the current environment
default_env = os.environ.copy()

def get_exec_prefix():
    exec_prefix = _shell_state.exec_prefix
    return sys.exec_prefix if exec_prefix == 'default' else exec_prefix

def get_conda_env():
    # what conda env am I in (e.g., where is my Python process from)?
    ENVBIN = get_exec_prefix()
    return Path(ENVBIN).name

# bind to 'Ctrl' + 'Space'
@bindings.add(Keys.ControlSpace)
def _(event):
    current_user_input = event.current_buffer.document.text
    func = _shell_state.function_type(SHELL_CONTEXT)

    bottom_toolbar = HTML(' <b>[f]</b> Print "f" <b>[x]</b> Abort.')

    # Create custom key bindings first.
    kb = KeyBindings()

    cancel = [False]
    @kb.add('f')
    def _(_event):
        UserMessage('You pressed `f`.', style="alert")

    @kb.add('x')
    def _(_event):
        " Send Abort (control-c) signal. "
        cancel[0] = True
        os.kill(os.getpid(), signal.SIGINT)

    # Use `patch_stdout`, to make sure that prints go above the
    # application.
    with patch_stdout(), ProgressBar(key_bindings=kb, bottom_toolbar=bottom_toolbar) as pb:
        # TODO: hack to simulate progress bar of indeterminate length of an synchronous function
        for i in pb(range(100)):
            if i > 50 and i < 70:
                time.sleep(.01)

            if i == 60:
                res = func(current_user_input) # hack to see progress bar

            # Stop when the cancel flag has been set.
            if cancel[0]:
                break

    with ConsoleStyle('code') as console:
        console.print(res)

@bindings.add(Keys.PageUp)
def _(event):
    # Moving up for 5 lines
    for _ in range(5):
        event.current_buffer.auto_up()

@bindings.add(Keys.PageDown)
def _(event):
    # Moving down for 5 lines
    for _ in range(5):
        event.current_buffer.auto_down()

class FileHistory(History):
    '''
    :class:`.History` class that stores all strings in a file.
    '''

    def __init__(self, filename: str) -> None:
        self.filename = Path(filename)
        super().__init__()

    def load_history_strings(self) -> Iterable[str]:
        lines: list[str] = []

        if self.filename.exists():
            with self.filename.open() as f:
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
        with self.filename.open("ab") as f:

            def write(t: str) -> None:
                f.write(t.encode("utf-8"))

            for line in string.split("\n"):
                write(f"{line}\n")

# Defining commands history
def load_history(home_path=HOME_PATH, history_file='.bash_history'):
    history_file_path = home_path / history_file
    history = FileHistory(history_file_path)
    return history, list(history.load_history_strings())

# Function to check if current directory is a git directory
def get_git_branch():
    try:
        git_process = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _stderr = git_process.communicate()
        if git_process.returncode == 0:
            return stdout.strip().decode('utf-8')
    except FileNotFoundError:
        pass
    return None

def disambiguate(cmds: str) -> tuple[str, int]:
    '''
    Ok, so, possible options for now:
        1. query | cmd
        2. query | file [file ...]
        -- not supported
        3. query | cmd | file
        4. query | cmd cmd ...
        5. query | file | cmd
    '''
    has_at_least_one_cmd = any(shutil.which(cmd) is not None for cmd in cmds.split(' '))
    maybe_cmd   = cmds.split(' ')[0].strip() # get first command
    maybe_files = FileReader.extract_files(cmds)
    # if cmd follows file(s) or file(s) follows cmd throw error as not supported
    if maybe_files is not None and has_at_least_one_cmd:
        msg = (
            'Cannot disambiguate commands that have both files and commands or multiple commands. Please provide '
            'correct order of commands. Supported are: query | file [file ...] (e.g. "what do these files have in '
            'common?" | file1 [file2 ...]) and query | cmd (e.g. "what flags can I use with rg?" | rg --help)'
        )
        UserMessage(msg, raise_with=ValueError)
    # now check order of commands and keep correct order
    if shutil.which(maybe_cmd) is not None:
        cmd_out = subprocess.run(cmds, check=False, capture_output=True, text=True, shell=True)
        if not cmd_out.stdout:
            msg = f'Command not found or failed. Error: {cmd_out.stderr}'
            UserMessage(msg, raise_with=ValueError)
        return cmd_out.stdout, 1
    if maybe_files is not None:
        return maybe_files, 2
    return None

# query language model
def _starts_with_prefix(query: str, prefix: str) -> bool:
    return (
        query.startswith(f'{prefix}"')
        or query.startswith(f"{prefix}'")
        or query.startswith(f'{prefix}`')
    )


def _is_new_conversation_query(query: str) -> bool:
    return _starts_with_prefix(query, '!')


def _is_followup_conversation_query(query: str) -> bool:
    return _starts_with_prefix(query, '.')


def _is_stateful_query(query: str) -> bool:
    return any(_starts_with_prefix(query, prefix) for prefix in ['.', '!'])


def _extract_query_kwargs(query: str, previous_kwargs, existing_kwargs):
    if '--kwargs' not in query and '-kw' not in query:
        return query, existing_kwargs, previous_kwargs

    splitter = '--kwargs' if '--kwargs' in query else '-kw'
    splits = query.split(splitter)
    suffix = splits[-1]
    if previous_kwargs is None and '=' not in suffix and ',' not in suffix:
        msg = 'Kwargs format must be last in query.'
        UserMessage(msg, raise_with=ValueError)
    if previous_kwargs is not None and '=' not in suffix and ',' not in suffix:
        cmd_kwargs = previous_kwargs
    else:
        query = splits[0].strip()
        kwargs_str = suffix.strip()
        cmd_kwargs = dict([kw.split('=') for kw in kwargs_str.split(',')])
        cmd_kwargs = {k.strip(): Symbol(v.strip()).ast() for k, v in cmd_kwargs.items()}

    previous_kwargs = cmd_kwargs
    merged_kwargs = {**existing_kwargs, **cmd_kwargs}
    return query, merged_kwargs, previous_kwargs


def _process_new_conversation(query, conversation_cls, symai_path, plugin, previous_kwargs, state):
    symai_path.parent.mkdir(parents=True, exist_ok=True)
    conversation = conversation_cls(auto_print=False)
    conversation_cls.save_conversation_state(conversation, symai_path)
    state.stateful_conversation = conversation
    if plugin is None:
        return conversation, previous_kwargs, None, False
    with Loader(desc="Inference ...", end=""):
        cmd = query[1:].strip('\'"')
        cmd = f"symrun {plugin} '{cmd}' --disable-pbar"
        cmd_out = run_shell_command(cmd, auto_query_on_error=True)
        conversation.store(cmd_out)
        conversation_cls.save_conversation_state(conversation, symai_path)
        state.stateful_conversation = conversation
        state.previous_kwargs = previous_kwargs
        return conversation, previous_kwargs, cmd_out, True


def _process_followup_conversation(query, conversation, conversation_cls, symai_path, plugin, previous_kwargs, state):
    try:
        conversation = conversation.load_conversation_state(symai_path)
        state.stateful_conversation = conversation
    except Exception:
        with ConsoleStyle('error') as console:
            console.print('No conversation state found. Please start a new conversation.')
        return conversation, previous_kwargs, None, True
    if plugin is None:
        return conversation, previous_kwargs, None, False
    with Loader(desc="Inference ...", end=""):
        trimmed_query = query[1:].strip('\'"')
        answer = conversation(trimmed_query).value
        cmd = f"symrun {plugin} '{answer}' --disable-pbar"
        cmd_out = run_shell_command(cmd, auto_query_on_error=True)
        conversation.store(cmd_out)
        conversation_cls.save_conversation_state(conversation, symai_path)
        state.stateful_conversation = conversation
        state.previous_kwargs = previous_kwargs
        return conversation, previous_kwargs, cmd_out, True


def _handle_piped_query(query, conversation, state):
    cmds = query.split('|')
    if len(cmds) > 2:
        msg = (
            'Cannot disambiguate commands that have more than 1 pipes. Please provide correct order of commands. '
            'Supported are: query | file [file ...] (e.g. "what do these files have in common?" | file1 [file2 ...]) '
            'and query | cmd (e.g. "what flags can I use with rg?" | rg --help)'
        )
        UserMessage(msg, raise_with=ValueError)
    base_query = cmds[0]
    payload, order = disambiguate(cmds[1].strip())
    is_stateful = _is_stateful_query(base_query)
    if is_stateful:
        func = conversation
    else:
        func = (
            state.function_type(payload)
            if order == 1
            else state.conversation_type(file_link=payload, auto_print=False)
        )
    if is_stateful:
        if order == 1:
            func.store(payload)
        elif order == 2:
            for file in payload:
                func.store_file(file)
    return func, base_query


def _select_function_for_query(query, conversation, state):
    if '|' in query:
        return _handle_piped_query(query, conversation, state)
    if _is_stateful_query(query):
        return conversation, query
    return state.function_type(SHELL_CONTEXT), query


def _should_save_conversation(conversation, query):
    if conversation is None:
        return False
    return _is_stateful_query(query)


def query_language_model(query: str, res=None, *args, **kwargs):
    state = _shell_state
    conversation = state.stateful_conversation
    previous_kwargs = state.previous_kwargs
    conversation_cls = state.conversation_type
    home_path = HOME_PATH
    symai_path = home_path / '.conversation_state'
    plugin = SYMSH_CONFIG.get('plugin_prefix')

    query, kwargs, previous_kwargs = _extract_query_kwargs(query, previous_kwargs, kwargs)

    if _is_new_conversation_query(query):
        conversation, previous_kwargs, result, should_return = _process_new_conversation(
            query, conversation_cls, symai_path, plugin, previous_kwargs, state
        )
        if should_return:
            return result
    elif _is_followup_conversation_query(query):
        conversation, previous_kwargs, result, should_return = _process_followup_conversation(
            query, conversation, conversation_cls, symai_path, plugin, previous_kwargs, state
        )
        if should_return:
            return result
    func, query = _select_function_for_query(query, conversation, state)
    with Loader(desc="Inference ...", end=""):
        query_to_execute = query
        if res is not None:
            query_to_execute = f"[Context]\n{res}\n\n[Query]\n{query}"
        msg = func(query_to_execute, *args, **kwargs)
        if res is not None:
            query = query_to_execute

    if _should_save_conversation(conversation, query):
        conversation_cls.save_conversation_state(conversation, symai_path)

    state.stateful_conversation = conversation
    state.previous_kwargs = previous_kwargs
    return msg

def retrieval_augmented_indexing(query: str, index_name = None, *_args, **_kwargs):
    state = _shell_state
    sep = os.path.sep
    path = query
    home_path = HOME_PATH

    # check if path contains overwrite flag
    overwrite = False
    if path.startswith('!'):
        overwrite = True
        path = path[1:]

    # check if request use of specific index
    use_index_name  = False
    if path.startswith('index:'):
        use_index_name = True
        # continue conversation with specific index
        index_name  = path.split('index:')[-1].strip()
    else:
        parse_arxiv = False

        # check if path contains arxiv flag
        if path.startswith('arxiv:'):
            parse_arxiv = True

        # check if path contains git flag
        if path.startswith('git@'):
            overwrite = True
            repo_path = home_path / 'temp'
            with Loader(desc="Cloning repo ...", end=""):
                cloner = RepositoryCloner(repo_path=str(repo_path))
                url = path[4:]
                if 'http' not in url:
                    url = 'https://' + url
                url = url.replace('.com:', '.com/')
                # if ends with '.git' then remove it
                if url.endswith('.git'):
                    url = url[:-4]
                path = cloner(url)

        # merge files
        merger = FileMerger()
        file = merger(path)
        # check if file contains arxiv pdf file and parse it
        if parse_arxiv:
            arxiv = ArxivPdfParser()
            pdf_file = arxiv(file)
            if pdf_file is not None:
                file = file |'\n'| pdf_file

        index_name = path.split(sep)[-1] if index_name is None else index_name
        index_name = Indexer.replace_special_chars(index_name)
        UserMessage(f'Indexing {index_name} ...', style="extensity")

        # creates index if not exists
        DocumentRetriever(index_name=index_name, file=file, overwrite=overwrite)

    symai_path = home_path / '.conversation_state'
    symai_path.parent.mkdir(parents=True, exist_ok=True)
    stateful_conversation = state.retrieval_conversation_type(auto_print=False, index_name=index_name)
    state.stateful_conversation = stateful_conversation
    Conversation.save_conversation_state(stateful_conversation, symai_path)
    if use_index_name:
        message = 'New session '
    else:
        message = f'Repository {url} cloned and ' if query.startswith('git@') or query.startswith('git:') else f'Directory {path} '
    return f'{message}successfully indexed: {index_name}'

def search_engine(query: str, res=None, *_args, **_kwargs):
    search = Interface('serpapi')
    with Loader(desc="Searching ...", end=""):
        search_query = Symbol(query).extract('search engine optimized query')
        res = search(search_query)
    with Loader(desc="Inference ...", end=""):
        func = _shell_state.function_type(query)
        msg = func(res, payload=res)
        # write a temp dump file with the query and results
        home_path = HOME_PATH
        symai_path = home_path / '.search_dump'
        symai_path.parent.mkdir(parents=True, exist_ok=True)
        with symai_path.open('w') as f:
            f.write(f'[SEARCH_QUERY]:\n{search_query}\n[RESULTS]\n{res}\n[MESSAGE]\n{msg}')
    return msg

def set_default_module(cmd: str):
    if cmd.startswith('set-plugin'):
        module = cmd.split('set-plugin')[-1].strip()
        SYMSH_CONFIG['plugin_prefix'] = module
        with config_path.open('w') as f:
            json.dump(SYMSH_CONFIG, f, indent=4)
        msg = f"Default plugin set to '{module}'"
    elif cmd == 'unset-plugin':
        SYMSH_CONFIG['plugin_prefix'] = None
        with config_path.open('w') as f:
            json.dump(SYMSH_CONFIG, f, indent=4)
        msg = "Default plugin unset"
    elif cmd == 'get-plugin':
        msg = f"Default plugin is '{SYMSH_CONFIG['plugin_prefix']}'"

    with ConsoleStyle('success') as console:
        console.print(msg)

def handle_error(cmd, res, message, auto_query_on_error):
    msg = Symbol(cmd) | f'\n{res!s}'
    if 'command not found' in str(res) or 'not recognized as an internal or external command' in str(res):
        return res.stderr.decode('utf-8')
    stderr = res.stderr
    if stderr and auto_query_on_error:
        rsp = stderr.decode('utf-8')
        UserMessage(rsp, style="alert")
        msg = msg | f"\n{rsp}"
        if 'usage:' in rsp:
            try:
                cmd = cmd.split('usage: ')[-1].split(' ')[0]
                # get man page result for command
                res = subprocess.run(f'man -P cat {cmd}',
                                        check=False, shell=True,
                                        stdout=subprocess.PIPE)
                stdout = res.stdout
                if stdout:
                    rsp = stdout.decode('utf-8')[:500]
                    msg = msg | f"\n{rsp}"
            except Exception:
                pass

        return query_language_model(msg)
    stdout = res.stdout
    if stdout:
        message = stderr.decode('utf-8')
    return message

# run shell command
def run_shell_command(cmd: str, prev=None, auto_query_on_error: bool=False, stdout=None, stderr=None):
    if prev is not None:
        cmd = prev + ' && ' + cmd
    message = None
    conda_env = get_exec_prefix()
    # copy default_env
    new_env = default_env.copy()
    if _shell_state.exec_prefix != 'default':
        # remove current env from PATH
        new_env["PATH"] = new_env["PATH"].replace(sys.exec_prefix, conda_env)
    # Execute the command
    try:
        stdout = subprocess.PIPE if auto_query_on_error else stdout
        stderr = subprocess.PIPE if auto_query_on_error else stderr
        res = subprocess.run(cmd, check=False, shell=True, stdout=stdout, stderr=stderr, env=new_env)
        if res and stdout and res.stdout:
            message = res.stdout.decode('utf-8')
        elif res and stderr and res.stderr:
            message = res.stderr.decode('utf-8')
    except FileNotFoundError as e:
        return e
    except PermissionError as e:
        return e

    # all good
    if res.returncode == 0:
        return message
    # If command not found, then try to query language model
    return handle_error(cmd, res, message, auto_query_on_error)

def is_llm_request(cmd: str):
    return cmd.startswith('"') or cmd.startswith('."') or cmd.startswith('!"') or cmd.startswith('?"') or\
           cmd.startswith("'") or cmd.startswith(".'") or cmd.startswith("!'") or cmd.startswith("?'") or\
           cmd.startswith('`') or cmd.startswith('.`') or cmd.startswith('!`') or cmd.startswith('?`') or\
           cmd.startswith('!(')

def map_nt_cmd(cmd: str, map_nt_cmd_enabled: bool = True):
    if os.name.lower() == 'nt' and map_nt_cmd_enabled and not is_llm_request(cmd):
        # Mapping command replacements with regex for commands with variants
        cmd_mappings = {
            r'\bls\b(-[a-zA-Z]*)?'         : r'dir \1',            # Maps 'ls' with or without arguments
            r'\bmv\b\s+(.*)'               : r'move \1',           # Maps 'mv' with any arguments
            r'\bcp\b\s+(.*)'               : r'copy \1',           # Maps 'cp' with any arguments
            r'\btouch\b\s+(.*)'            : r'type nul > \1',     # Maps 'touch filename' to 'type nul > filename'
            r'\brm\b\s+(-rf)?'             : r'del \1',            # Maps 'rm' and 'rm -rf'
            r'\bdiff\b\s+(.*)'             : r'fc \1',             # Maps 'diff' with any arguments
            r'\bgrep\b\s+(.*)'             : r'find \1',           # Maps 'grep' with any arguments
            r'\bpwd\b'                     : 'chdir',              # pwd has no arguments
            r'\bdate\b'                    : 'time',               # date has no arguments
            r'\bmkdir\b\s+(.*)'            : r'md \1',             # Maps 'mkdir' with any arguments
            r'\bwhich\b\s+(.*)'            : r'where \1',          # Maps 'which' with any arguments
            r'\b(vim|nano)\b\s+(.*)'       : r'notepad \2',        # Maps 'vim' or 'nano' with any arguments
            r'\b(mke2fs|mformat)\b\s+(.*)' : r'format \2',         # Maps 'mke2fs' or 'mformat' with any arguments
            r'\b(rm\s+-rf|rmdir)\b'        : 'rmdir /s /q',        # Matches 'rm -rf' or 'rmdir'
            r'\bkill\b\s+(.*)'             : r'taskkill \1',       # Maps 'kill' with any arguments
            r'\bps\b\s*(.*)?'              : r'tasklist \1',       # Maps 'ps' with any or no arguments
            r'\bexport\b\s+(.*)'           : r'set \1',            # Maps 'export' with any arguments
            r'\b(chown|chmod)\b\s+(.*)'    : r'attrib +r \2',      # Maps 'chown' or 'chmod' with any arguments
            r'\btraceroute\b\s+(.*)'       : r'tracert \1',        # Maps 'traceroute' with any arguments
            r'\bcron\b\s+(.*)'             : r'at \1',             # Maps 'cron' with any arguments
            r'\bcat\b\s+(.*)'              : r'type \1',           # Maps 'cat' with any arguments
            r'\bdu\s+-s\b'                 : 'chkdsk',             # du -s has no arguments, chkdsk is closest in functionality
            r'\bls\s+-R\b'                 : 'tree',               # ls -R has no arguments
        }

        # Remove 1:1 mappings
        direct_mappings = {
            'clear': 'cls',
            'man'  : 'help',
            'mem'  : 'free',
        }

        cmd_mappings.update(direct_mappings)

        # Iterate through mappings and replace commands
        for linux_cmd, windows_cmd in cmd_mappings.items():
            # copy original command
            original_cmd = cmd
            cmd = re.sub(linux_cmd, windows_cmd, cmd)
            if cmd != original_cmd:
                UserMessage(f'symsh >> command "{original_cmd}" mapped to "{cmd}"\n', style="extensity")

    return cmd


def _handle_plugin_commands(cmd: str):
    if cmd.startswith('set-plugin') or cmd == 'unset-plugin' or cmd == 'get-plugin':
        return set_default_module(cmd)
    return None


def _handle_chained_llm_commands(cmd: str, res, auto_query_on_error: bool):
    if '" && ' not in cmd and "' && " not in cmd and '` && ' not in cmd:
        return None
    if not is_llm_request(cmd):
        return run_shell_command(cmd, prev=res, auto_query_on_error=auto_query_on_error)
    cmds = cmd.split(' && ')
    if not is_llm_request(cmds[0]):
        return ValueError('The first command must be a LLM request.')
    first_res = query_language_model(cmds[0], res=res)
    rest = ' && '.join(cmds[1:])
    if len(cmds) > 1 and '$1' in cmds[1]:
        first_res_str = str(first_res).replace('\n', r'\\n')
        rest = rest.replace('$1', f'"{first_res_str}"')
        first_res = None
    return run_shell_command(rest, prev=first_res, auto_query_on_error=auto_query_on_error)


def _handle_llm_or_search(cmd: str, res):
    if cmd.startswith('?"') or cmd.startswith("?'") or cmd.startswith('?`'):
        query = cmd[1:]
        return search_engine(query, res=res)
    if is_llm_request(cmd) or '...' in cmd:
        return query_language_model(cmd, res=res)
    return None


def _handle_retrieval_commands(cmd: str):
    if cmd.startswith('*'):
        return retrieval_augmented_indexing(cmd[1:])
    return None


def _handle_man_command(cmd: str):
    if cmd.startswith('man symsh'):
        pkg_path = Path(__file__).resolve().parent
        symsh_path = pkg_path / 'symsh.md'
        with symsh_path.open(encoding="utf8") as file_ptr:
            return file_ptr.read()
    return None


def _handle_conda_commands(cmd: str, state, res, auto_query_on_error: bool):
    if cmd.startswith('conda activate'):
        env = Path(sys.exec_prefix)
        env_base = env.parent
        req_env = cmd.split(' ')[2]
        env_path = env_base / req_env
        if not env_path.exists():
            return f'Environment {req_env} does not exist!'
        state.previous_prefix = state.exec_prefix
        state.exec_prefix = str(env_path)
        return state.exec_prefix
    if cmd.startswith('conda deactivate'):
        prev_prefix = state.previous_prefix
        if prev_prefix is not None:
            state.exec_prefix = prev_prefix
        if prev_prefix == 'default':
            state.previous_prefix = None
        return get_exec_prefix()
    if cmd.startswith('conda'):
        env = Path(get_exec_prefix())
        try:
            env_base = env.parents[1]
        except IndexError:
            env_base = env.parent
        cmd_rewritten = cmd.replace('conda', str(env_base / "condabin" / "conda"))
        return run_shell_command(cmd_rewritten, prev=res, auto_query_on_error=auto_query_on_error)
    return None


def _handle_directory_navigation(cmd: str):
    sep = os.path.sep
    if cmd.startswith('cd'):
        try:
            cmd_expanded = FileReader.expand_user_path(cmd)
            path = ' '.join(cmd_expanded.split(' ')[1:])
            if path.endswith(sep):
                path = path[:-1]
            return os.chdir(path)
        except FileNotFoundError as err:
            return err
        except PermissionError as err:
            return err
    cmd_path = FileReader.expand_user_path(cmd)
    if Path(cmd).is_dir():
        try:
            os.chdir(cmd_path)
        except FileNotFoundError as err:
            return err
        except PermissionError as err:
            return err
    return None


def _handle_ll_alias(cmd: str, res):
    if not cmd.startswith('ll'):
        return None
    if os.name == 'nt':
        rewritten = cmd.replace('ll', 'dir')
        return run_shell_command(rewritten, prev=res)
    rewritten = cmd.replace('ll', 'ls -la')
    return run_shell_command(rewritten, prev=res)


def process_command(cmd: str, res=None, auto_query_on_error: bool=False):
    state = _shell_state

    # map commands to windows if needed
    cmd = map_nt_cmd(cmd)
    plugin_result = _handle_plugin_commands(cmd)
    if plugin_result is not None:
        return plugin_result

    chained_result = _handle_chained_llm_commands(cmd, res, auto_query_on_error)
    if chained_result is not None:
        return chained_result

    llm_or_search = _handle_llm_or_search(cmd, res)
    if llm_or_search is not None:
        return llm_or_search

    retrieval_result = _handle_retrieval_commands(cmd)
    if retrieval_result is not None:
        return retrieval_result

    man_result = _handle_man_command(cmd)
    if man_result is not None:
        return man_result

    conda_result = _handle_conda_commands(cmd, state, res, auto_query_on_error)
    if conda_result is not None:
        return conda_result

    directory_result = _handle_directory_navigation(cmd)
    if directory_result is not None:
        return directory_result

    ll_result = _handle_ll_alias(cmd, res, auto_query_on_error)
    if ll_result is not None:
        return ll_result

    return run_shell_command(cmd, prev=res, auto_query_on_error=auto_query_on_error)

def save_conversation():
    home_path = HOME_PATH
    symai_path = home_path / '.conversation_state'
    Conversation.save_conversation_state(_shell_state.stateful_conversation, symai_path)


def _is_exit_command(cmd: str) -> bool:
    return cmd in ['quit', 'exit', 'q']


def _format_working_directory():
    sep = os.path.sep
    cur_working_dir = Path.cwd()
    cur_working_dir_str = str(cur_working_dir)
    if cur_working_dir_str.startswith(sep):
        cur_working_dir_str = FileReader.expand_user_path(cur_working_dir_str)
    paths = cur_working_dir_str.split(sep)
    prev_paths = sep.join(paths[:-1])
    last_path = paths[-1]
    if len(paths) > 1:
        return f'{prev_paths}{sep}<b>{last_path}</b>'
    return f'<b>{last_path}</b>'


def _build_prompt(git_branch, conda_env, cur_working_dir_str):
    if git_branch:
        return HTML(
            f"<ansiblue>{cur_working_dir_str}</ansiblue><ansiwhite> on git:[</ansiwhite>"
            f"<ansigreen>{git_branch}</ansigreen><ansiwhite>]</ansiwhite> <ansiwhite>conda:[</ansiwhite>"
            f"<ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ></ansicyan> "
        )
    return HTML(
        f"<ansiblue>{cur_working_dir_str}</ansiblue> <ansiwhite>conda:[</ansiwhite>"
        f"<ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ></ansicyan> "
    )


def _handle_exit(state):
    if state.stateful_conversation is not None:
        save_conversation()
    if not state.use_styles:
        UserMessage('Goodbye!', style="extensity")
    else:
        func = _shell_state.function_type('Give short goodbye')
        UserMessage(func('bye'), style="extensity")
    os._exit(0)


# Function to listen for user input and execute commands
def listen(session: PromptSession, word_comp: WordCompleter, auto_query_on_error: bool=False, verbose: bool=False):
    state = _shell_state
    with patch_stdout():
        while True:
            try:
                git_branch = get_git_branch()
                conda_env = get_conda_env()
                cur_working_dir_str = _format_working_directory()
                prompt = _build_prompt(git_branch, conda_env, cur_working_dir_str)
                cmd = session.prompt(prompt)
                if cmd.strip() == '':
                    continue

                if _is_exit_command(cmd):
                    _handle_exit(state)
                msg = process_command(cmd, auto_query_on_error=auto_query_on_error)
                if msg is not None:
                    with ConsoleStyle('code') as console:
                        console.print(msg)

                # Append the command to the word completer list
                word_comp.words.append(cmd)

            except KeyboardInterrupt:
                UserMessage('', style="alert")
            except Exception as e:
                UserMessage(str(e), style="alert")
                if verbose:
                    traceback.print_exc()

def create_session(history, merged_completer):
    colors = SYMSH_CONFIG['colors']

    # Load style
    style = Style.from_dict(colors)

    # Session for the auto-completion
    return PromptSession(history=history,
                         completer=merged_completer,
                         complete_style=CompleteStyle.MULTI_COLUMN,
                         reserve_space_for_menu=5,
                         style=style,
                         key_bindings=bindings)

def create_completer():
    # Load history
    history, history_strings = load_history()

    # Create your specific completers
    word_comp = HistoryCompleter(history_strings, ignore_case=True, sentence=True)
    custom_completer = PathCompleter()

    # Merge completers
    merged_completer = MergedCompleter(custom_completer, word_comp)
    return history, word_comp, merged_completer

def run(auto_query_on_error=False, conversation_style=None, verbose=False):
    state = _shell_state
    if conversation_style is not None and conversation_style != '':
        UserMessage(f'Loading style: {conversation_style}', style="extensity")
        styles_ = Import.load_module_class(conversation_style)
        (
            state.function_type,
            state.conversation_type,
            state.retrieval_conversation_type,
        ) = styles_
        state.use_styles = True

    if SYMSH_CONFIG['show-splash-screen']:
        show_intro_menu()
        # set show splash screen to false
        SYMSH_CONFIG['show-splash-screen'] = False
        # save config
        _config_path = HOME_PATH / 'symsh.config.json'
        with _config_path.open('w') as f:
            json.dump(SYMSH_CONFIG, f, indent=4)
    if 'plugin_prefix' not in SYMSH_CONFIG:
        SYMSH_CONFIG['plugin_prefix'] = None

    history, word_comp, merged_completer = create_completer()
    session = create_session(history, merged_completer)
    listen(session, word_comp, auto_query_on_error=auto_query_on_error, verbose=verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SymSH: Symbolic Shell')
    parser.add_argument('--auto-query-on-error', action='store_true', help='Automatically query the language model on error.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose errors.')
    args = parser.parse_args()
    run(auto_query_on_error=args.auto_query_on_error, conversation_style=args.conversation_style, verbose=args.verbose)
