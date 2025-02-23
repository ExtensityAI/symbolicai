import argparse
import glob
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
from pathlib import Path
from typing import Iterable, Tuple

from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import CompleteStyle, ProgressBar
from prompt_toolkit.styles import Style
from pygments.lexers.shell import BashLexer

from .backend.settings import HOME_PATH, SYMSH_CONFIG
from .components import FileReader, Function, Indexer
from .extended import (ArxivPdfParser, Conversation, DocumentRetriever,
                       FileMerger, RepositoryCloner,
                       RetrievalAugmentedConversation)
from .imports import Import
from .interfaces import Interface
from .menu.screen import show_intro_menu
from .misc.console import ConsoleStyle
from .misc.loader import Loader
from .symbol import Symbol

logging.getLogger("prompt_toolkit").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("subprocess").setLevel(logging.ERROR)

# load json config from home directory root
home_path = HOME_PATH
config_path = os.path.join(home_path, 'symsh.config.json')
# migrate config from old path
if 'colors' not in SYMSH_CONFIG:
    __new_config__ = {"colors": SYMSH_CONFIG}
    # add command in config
    SYMSH_CONFIG = __new_config__
    # save config
    with open(config_path, 'w') as f:
        json.dump(__new_config__, f, indent=4)

# make sure map-nt-cmd is in config
if 'map-nt-cmd' not in SYMSH_CONFIG:
    # add command in config
    SYMSH_CONFIG['map-nt-cmd'] = True
    # save config
    with open(config_path, 'w') as f:
        json.dump(SYMSH_CONFIG, f, indent=4)

print                     = print_formatted_text
FunctionType              = Function
ConversationType          = Conversation
RetrievalConversationType = RetrievalAugmentedConversation
use_styles                = False
map_nt_cmd_enabled        = SYMSH_CONFIG['map-nt-cmd']

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

stateful_conversation = None
previous_kwargs       = None

def supports_ansi_escape():
    try:
        os.get_terminal_size(0)
        return True
    except OSError:
        return False

class PathCompleter(Completer):
    def get_completions(self, document, complete_event):
        complete_word = document.get_word_before_cursor(WORD=True)
        sep = os.path.sep
        if complete_word.startswith(f'~{sep}'):
            complete_word = FileReader.expand_user_path(complete_word)

        # list all files and directories in current directory
        files = list(glob.glob(complete_word + '*'))
        if len(files) == 0:
            return None

        dirs_ = []
        files_ = []

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
            if os.path.isdir(file):
                dirs_.append(file)
            else:
                files_.append(file)

        for d in dirs_:
            # if starts with home directory, then replace it with ~
            d = FileReader.expand_user_path(d)
            yield Completion(d, start_position=-start_position,
                             style='class:path-completion',
                             selected_style='class:path-completion-selected')

        for f in files_:
            # if starts with home directory, then replace it with ~
            f = FileReader.expand_user_path(f)
            yield Completion(f, start_position=-start_position,
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
previous_prefix = None
exec_prefix = 'default'
# Get a copy of the current environment
default_env = os.environ.copy()

def get_exec_prefix():
    return sys.exec_prefix if exec_prefix == 'default' else exec_prefix

def get_conda_env():
    # what conda env am I in (e.g., where is my Python process from)?
    ENVBIN = get_exec_prefix()
    env_name = os.path.basename(ENVBIN)
    return env_name

# bind to 'Ctrl' + 'Space'
@bindings.add(Keys.ControlSpace)
def _(event):
    current_user_input = event.current_buffer.document.text
    func = FunctionType(SHELL_CONTEXT)

    bottom_toolbar = HTML(' <b>[f]</b> Print "f" <b>[x]</b> Abort.')

    # Create custom key bindings first.
    kb = KeyBindings()

    cancel = [False]
    @kb.add('f')
    def _(event):
        print('You pressed `f`.')

    @kb.add('x')
    def _(event):
        " Send Abort (control-c) signal. "
        cancel[0] = True
        os.kill(os.getpid(), signal.SIGINT)

    # Use `patch_stdout`, to make sure that prints go above the
    # application.
    with patch_stdout():
        with ProgressBar(key_bindings=kb, bottom_toolbar=bottom_toolbar) as pb:
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
def load_history(home_path=HOME_PATH, history_file='.bash_history'):
    history_file_path = os.path.join(home_path, history_file)
    history = FileHistory(history_file_path)
    return history, list(history.load_history_strings())

# Function to check if current directory is a git directory
def get_git_branch():
    try:
        git_process = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = git_process.communicate()
        if git_process.returncode == 0:
            return stdout.strip().decode('utf-8')
    except FileNotFoundError:
        pass
    return None

def disambiguate(cmds: str) -> Tuple[str, int]:
    '''
    Ok, so, possible options for now:
        1. query | cmd
        2. query | file [file ...]
        -- not supported
        3. query | cmd | file
        4. query | cmd cmd ...
        5. query | file | cmd
    '''
    has_at_least_one_cmd = any([shutil.which(cmd) is not None for cmd in cmds.split(' ')])
    maybe_cmd   = cmds.split(' ')[0].strip() # get first command
    maybe_files = FileReader.extract_files(cmds)
    # if cmd follows file(s) or file(s) follows cmd throw error as not supported
    if maybe_files is not None and has_at_least_one_cmd:
        raise ValueError('Cannot disambiguate commands that have both files and commands or multiple commands. Please provide correct order of commands. '
                         'Supported are: '
                         'query | file [file ...] (e.g. "what do these files have in common?" | file1 [file2 ...]) '
                         'and '
                         'query | cmd (e.g. "what flags can I use with rg?" | rg --help)')
    # now check order of commands and keep correct order
    if shutil.which(maybe_cmd) is not None:
        cmd_out = subprocess.run(cmds, capture_output=True, text=True, shell=True)
        if not cmd_out.stdout:
            raise ValueError(f'Command not found or failed. Error: {cmd_out.stderr}')
        return cmd_out.stdout, 1
    if maybe_files is not None:
        return maybe_files, 2

# query language model
def query_language_model(query: str, res=None, *args, **kwargs):
    global stateful_conversation, previous_kwargs
    home_path = HOME_PATH
    symai_path = os.path.join(home_path, '.conversation_state')
    plugin = SYMSH_CONFIG.get('plugin_prefix')

    # check and extract kwargs from query if any
    # format --kwargs key1=value1,key2=value2,key3=value3,...keyN=valueN
    if '--kwargs' in query or '-kw' in query:
        splitter = '--kwargs' if '--kwargs' in query else '-kw'
        # check if kwargs format is last in query otherwise raise error
        splits = query.split(f'{splitter}')
        if previous_kwargs is None and '=' not in splits[-1] and ',' not in splits[-1]:
            raise ValueError('Kwargs format must be last in query.')
        elif previous_kwargs is not None and '=' not in splits[-1] and ',' not in splits[-1]:
            # use previous kwargs
            cmd_kwargs = previous_kwargs
        else:
            # remove kwargs from query
            query = splits[0].strip()
            # extract kwargs
            kwargs_str = splits[-1].strip()
            cmd_kwargs = dict([kw.split('=') for kw in kwargs_str.split(',')])
            cmd_kwargs = {k.strip(): Symbol(v.strip()).ast() for k, v in cmd_kwargs.items()}

        previous_kwargs = cmd_kwargs
        # unpack cmd_kwargs to kwargs
        kwargs = {**kwargs, **cmd_kwargs}

    # Handle stateful conversations:
    # 1. If query starts with !" (new conversation), create new conversation state
    # 2. If query starts with ." (follow-up), either:
    #    - Load existing conversation state if it exists
    #    - Create new conversation state if none exists
    if (query.startswith('!"') or query.startswith("!'") or query.startswith('!`')):
        os.makedirs(os.path.dirname(symai_path), exist_ok=True)
        stateful_conversation = ConversationType(auto_print=False)
        ConversationType.save_conversation_state(stateful_conversation, symai_path)
        # Special case: if query starts with !" and has a prefix, run the prefix command and store the output
        if plugin is not None:
            with Loader(desc="Inference ...", end=""):
                cmd = query[1:].strip('\'"')
                cmd = f"symrun {plugin} '{cmd}' --disable-pbar"
                cmd_out = run_shell_command(cmd, auto_query_on_error=True)
                stateful_conversation.store(cmd_out)
                ConversationType.save_conversation_state(stateful_conversation, symai_path)
                return cmd_out
    elif query.startswith('."') or query.startswith(".'") or query.startswith('.`'):
        try:
            stateful_conversation = stateful_conversation.load_conversation_state(symai_path)
        except Exception:
            with ConsoleStyle('error') as console:
                console.print('No conversation state found. Please start a new conversation.')
            return
        if plugin is not None:
            with Loader(desc="Inference ...", end=""):
                query = query[1:].strip('\'"')
                answer = stateful_conversation(query).value
                cmd = f"symrun {plugin} '{answer}' --disable-pbar"
                cmd_out = run_shell_command(cmd, auto_query_on_error=True)
                stateful_conversation.store(cmd_out)
                ConversationType.save_conversation_state(stateful_conversation, symai_path)
                return cmd_out
    cmd = None
    if '|' in query:
        cmds = query.split('|')
        if len(cmds) > 2:
            raise ValueError(('Cannot disambiguate commands that have more than 1 pipes. Please provide correct order of commands. '
                              'Supported are: '
                              'query | file [file ...] (e.g. "what do these files have in common?" | file1 [file2 ...]) '
                              'and '
                              'query | cmd (e.g. "what flags can I use with rg?" | rg --help)'))
        query = cmds[0]
        payload, order = disambiguate(cmds[1].strip())
        # check if we're in a stateful conversation
        is_stateful = query.startswith(('.', '!')) and any(query.startswith(f"{prefix}{quote}")
                     for prefix in ['.', '!']
                     for quote in ['"', "'", '`'])

        if is_stateful:
            func = stateful_conversation
        else:
            func = FunctionType(payload) if order == 1 else ConversationType(file_link=payload, auto_print=False)

        if is_stateful:
            if order == 1:
                func.store(payload)
            elif order == 2:
                for file in payload:
                    func.store_file(file)
    else:
        if  query.startswith('."') or query.startswith(".'") or query.startswith('.`') or\
            query.startswith('!"') or query.startswith("!'") or query.startswith('!`'):
            func = stateful_conversation
        else:
            func = FunctionType(SHELL_CONTEXT)

    with Loader(desc="Inference ...", end=""):
        if res is not None:
            query = f"[Context]\n{res}\n\n[Query]\n{query}"
        msg = func(query, *args, **kwargs)

    if stateful_conversation is not None and (
        query.startswith('."') or query.startswith(".'") or query.startswith('.`') or
        query.startswith('!"') or query.startswith("!'") or query.startswith('!`')
    ):
        ConversationType.save_conversation_state(stateful_conversation, symai_path)

    return msg

def retrieval_augmented_indexing(query: str, index_name = None, *args, **kwargs):
    global stateful_conversation
    sep = os.path.sep
    path = query

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
            repo_path = os.path.join(home_path, 'temp')
            cloner = RepositoryCloner(repo_path=repo_path)
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
        print(f'Indexing {index_name} ...')

        # creates index if not exists
        DocumentRetriever(index_name=index_name, file=file, overwrite=overwrite)

    home_path = HOME_PATH
    symai_path = os.path.join(home_path, '.conversation_state')
    os.makedirs(os.path.dirname(symai_path), exist_ok=True)
    stateful_conversation = RetrievalConversationType(auto_print=False, index_name=index_name)
    Conversation.save_conversation_state(stateful_conversation, symai_path)
    if use_index_name:
        message = 'New session '
    else:
        message = f'Repository {url} cloned and ' if query.startswith('git@') or query.startswith('git:') else f'Directory {path} '
    msg = f'{message}successfully indexed: {index_name}'
    return msg

def search_engine(query: str, res=None, *args, **kwargs):
    search = Interface('serpapi')
    with Loader(desc="Searching ...", end=""):
        search_query = Symbol(query).extract('search engine optimized query')
        res = search(search_query)
    with Loader(desc="Inference ...", end=""):
        func = FunctionType(query)
        msg = func(res, payload=res)
        # write a temp dump file with the query and results
        home_path = HOME_PATH
        symai_path = os.path.join(home_path, '.search_dump')
        os.makedirs(os.path.dirname(symai_path), exist_ok=True)
        with open(symai_path, 'w') as f:
            f.write(f'[SEARCH_QUERY]:\n{search_query}\n[RESULTS]\n{res}\n[MESSAGE]\n{msg}')
    return msg

def set_default_module(cmd: str):
    if cmd.startswith('set-plugin'):
        module = cmd.split('set-plugin')[-1].strip()
        SYMSH_CONFIG['plugin_prefix'] = module
        with open(config_path, 'w') as f:
            json.dump(SYMSH_CONFIG, f, indent=4)
        msg = f"Default plugin set to '{module}'"
    elif cmd == 'unset-plugin':
        SYMSH_CONFIG['plugin_prefix'] = None
        with open(config_path, 'w') as f:
            json.dump(SYMSH_CONFIG, f, indent=4)
        msg = "Default plugin unset"
    elif cmd == 'get-plugin':
        msg = f"Default plugin is '{SYMSH_CONFIG['plugin_prefix']}'"

    with ConsoleStyle('success') as console:
        console.print(msg)

def handle_error(cmd, res, message, auto_query_on_error):
    msg = Symbol(cmd) | f'\n{str(res)}'
    if 'command not found' in str(res) or 'not recognized as an internal or external command' in str(res):
        return res.stderr.decode('utf-8')
    else:
        stderr = res.stderr
        if stderr and auto_query_on_error:
            rsp = stderr.decode('utf-8')
            print(rsp)
            msg = msg | f"\n{rsp}"
            if 'usage:' in rsp:
                try:
                    cmd = cmd.split('usage: ')[-1].split(' ')[0]
                    # get man page result for command
                    res = subprocess.run('man -P cat %s' % cmd,
                                            shell=True,
                                            stdout=subprocess.PIPE)
                    stdout = res.stdout
                    if stdout:
                        rsp = stdout.decode('utf-8')[:500]
                        msg = msg | f"\n{rsp}"
                except Exception:
                    pass

            return query_language_model(msg)
        else:
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
    if exec_prefix != 'default':
        # remove current env from PATH
        new_env["PATH"] = new_env["PATH"].replace(sys.exec_prefix, conda_env)
    # Execute the command
    try:
        stdout = subprocess.PIPE if auto_query_on_error else stdout
        stderr = subprocess.PIPE if auto_query_on_error else stderr
        res = subprocess.run(cmd, shell=True, stdout=stdout, stderr=stderr, env=new_env)
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
    else:
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
                print(f'symsh >> command "{original_cmd}" mapped to "{cmd}"\n')

    return cmd

def process_command(cmd: str, res=None, auto_query_on_error: bool=False):
    global exec_prefix, previous_prefix

    # map commands to windows if needed
    cmd = map_nt_cmd(cmd)
    if cmd.startswith('set-plugin') or cmd == 'unset-plugin' or cmd == 'get-plugin':
        return set_default_module(cmd)

    sep = os.path.sep
    # check for '&&' to also preserve pipes '|' in normal shell commands
    if '" && ' in cmd or "' && " in cmd or '` && ' in cmd:
        if is_llm_request(cmd):
            # Process each command (the ones involving the LLM) separately
            cmds = cmd.split(' && ')
            if not is_llm_request(cmds[0]):
                return ValueError('The first command must be a LLM request.')
            # Process the first command as an LLM request
            res = query_language_model(cmds[0], res=res)
            rest = ' && '.join(cmds[1:])
            if '$1' in cmds[1]:
                res  = str(res).replace('\n', r'\\n')
                rest = rest.replace('$1', '"%s"' % res)
                res  = None
            cmd = rest
        # If it's a normal shell command with pipes or &&, pass it whole
        res = run_shell_command(cmd, prev=res, auto_query_on_error=auto_query_on_error)
        return res

    if cmd.startswith('?"') or cmd.startswith("?'") or cmd.startswith('?`'):
        cmd = cmd[1:]
        return search_engine(cmd, res=res)

    elif  is_llm_request(cmd) or '...' in cmd:
        return query_language_model(cmd, res=res)

    elif cmd.startswith('*'):
        cmd = cmd[1:]
        return retrieval_augmented_indexing(cmd)

    elif cmd.startswith('man symsh'):
        # read symsh.md file and print it
        # get symsh path
        pkg_path = os.path.dirname(os.path.abspath(__file__))
        symsh_path = os.path.join(pkg_path, 'symsh.md')
        with open(symsh_path, 'r', encoding="utf8") as f:
            return f.read()

    elif cmd.startswith('conda activate'):
        # check conda execution prefix and verify if environment exists
        env = sys.exec_prefix
        path_ = sep.join(env.split(sep)[:-1])
        env_base = os.path.join(sep, path_)
        req_env = cmd.split(' ')[2]
        # check if environment exists
        env_path = os.path.join(env_base, req_env)
        if not os.path.exists(env_path):
            return f'Environment {req_env} does not exist!'
        previous_prefix = exec_prefix
        exec_prefix = os.path.join(env_base, req_env)
        return exec_prefix

    elif cmd.startswith('conda deactivate'):
        if previous_prefix is not None:
            exec_prefix = previous_prefix
        if previous_prefix == 'default':
            previous_prefix = None
        return get_exec_prefix()

    elif cmd.startswith('conda'):
        env = get_exec_prefix()
        env_base = os.path.join(sep, *env.split(sep)[:-2])
        cmd = cmd.replace('conda', os.path.join(env_base, "condabin", "conda"))
        return run_shell_command(cmd, prev=res, auto_query_on_error=auto_query_on_error)

    elif cmd.startswith('cd'):
        try:
            # replace ~ with home directory
            cmd = FileReader.expand_user_path(cmd)
            # Change directory
            path = ' '.join(cmd.split(' ')[1:])
            if path.endswith(sep):
                path = path[:-1]
            return os.chdir(path)
        except FileNotFoundError as e:
            return e
        except PermissionError as e:
            return e

    elif os.path.isdir(cmd):
        try:
            # replace ~ with home directory
            cmd = FileReader.expand_user_path(cmd)
            # Change directory
            os.chdir(cmd)
        except FileNotFoundError as e:
            return e
        except PermissionError as e:
            return e

    elif cmd.startswith('ll'):

        if os.name == 'nt':
            cmd = cmd.replace('ll', 'dir')
            return run_shell_command(cmd, prev=res)
        else:
            cmd = cmd.replace('ll', 'ls -la')
            return run_shell_command(cmd, prev=res)

    else:
        return run_shell_command(cmd, prev=res, auto_query_on_error=auto_query_on_error)

def save_conversation():
    home_path = HOME_PATH
    symai_path = os.path.join(home_path, '.conversation_state')
    Conversation.save_conversation_state(stateful_conversation, symai_path)

# Function to listen for user input and execute commands
def listen(session: PromptSession, word_comp: WordCompleter, auto_query_on_error: bool=False, verbose: bool=False):
    with patch_stdout():
        while True:
            try:
                git_branch = get_git_branch()
                conda_env = get_conda_env()
                # get directory from the shell
                cur_working_dir = os.getcwd()
                sep = os.path.sep
                if cur_working_dir.startswith(sep):
                    cur_working_dir = FileReader.expand_user_path(cur_working_dir)
                paths = cur_working_dir.split(sep)
                prev_paths = sep.join(paths[:-1])
                last_path = paths[-1]

                # Format the prompt
                if len(paths) > 1:
                    cur_working_dir = f'{prev_paths}{sep}<b>{last_path}</b>'
                else:
                    cur_working_dir = f'<b>{last_path}</b>'

                if git_branch:
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue><ansiwhite> on git:[</ansiwhite><ansigreen>{git_branch}</ansigreen><ansiwhite>]</ansiwhite> <ansiwhite>conda:[</ansiwhite><ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ❯</ansicyan> ")
                else:
                    prompt = HTML(f"<ansiblue>{cur_working_dir}</ansiblue> <ansiwhite>conda:[</ansiwhite><ansimagenta>{conda_env}</ansimagenta><ansiwhite>]</ansiwhite> <ansicyan><b>symsh:</b> ❯</ansicyan> ")

                # Read user input
                cmd = session.prompt(prompt)
                if cmd.strip() == '':
                    continue

                if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                    if stateful_conversation is not None:
                        save_conversation()
                    if not use_styles:
                        print('Goodbye!')
                    else:
                        func = FunctionType('Give short goodbye')
                        print(func('bye'))
                    os._exit(0)
                else:
                    msg = process_command(cmd, auto_query_on_error=auto_query_on_error)
                    if msg is not None:
                        with ConsoleStyle('code') as console:
                            console.print(msg)

                # Append the command to the word completer list
                word_comp.words.append(cmd)

            except KeyboardInterrupt:
                print()
                pass

            except Exception as e:
                print(e)
                if verbose:
                    traceback.print_exc()
                pass

def create_session(history, merged_completer):
    colors = SYMSH_CONFIG['colors']

    # Load style
    style = Style.from_dict(colors)

    # Session for the auto-completion
    session = PromptSession(history=history,
                            completer=merged_completer,
                            complete_style=CompleteStyle.MULTI_COLUMN,
                            reserve_space_for_menu=5,
                            style=style,
                            key_bindings=bindings)

    return session

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
    global FunctionType, ConversationType, RetrievalConversationType, use_styles
    if conversation_style is not None and conversation_style != '':
        print('Loading style:', conversation_style)
        styles_ = Import.load_module_class(conversation_style)
        FunctionType, ConversationType, RetrievalConversationType = styles_
        use_styles = True

    if SYMSH_CONFIG['show-splash-screen']:
        show_intro_menu()
        # set show splash screen to false
        SYMSH_CONFIG['show-splash-screen'] = False
        # save config
        _config_path =  HOME_PATH / 'symsh.config.json'
        with open(_config_path, 'w') as f:
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
