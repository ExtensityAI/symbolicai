import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .backend import settings
from .menu.screen import show_menu
from .misc.console import ConsoleStyle
from .utils import CustomUserWarning

# do not remove - hides the libraries' debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("tika").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface").setLevel(logging.ERROR)
logging.getLogger('pydub').setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# set the environment variable for the transformers library
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# Create singleton instance
config_manager = settings.SymAIConfig()

SYMAI_VERSION = "0.13.2"
__version__   = SYMAI_VERSION
__root_dir__  = config_manager.config_dir

def _start_symai():
    global _symai_config_
    global _symsh_config_
    global _symserver_config_

    # Create config directories if they don't exist
    os.makedirs(config_manager._env_config_dir, exist_ok=True)
    os.makedirs(config_manager._home_config_dir, exist_ok=True)

    # CREATE THE SHELL CONFIGURATION FILE IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    _symsh_config_path_ = config_manager.get_config_path('symsh.config.json')
    if not os.path.exists(_symsh_config_path_):
        config_manager.save_config('symsh.config.json', {
            "colors": {
                "completion-menu.completion.current": "bg:#323232 #212121",
                "completion-menu.completion":         "bg:#800080 #212121",
                "scrollbar.background":               "bg:#222222",
                "scrollbar.button":                   "bg:#776677",
                "history-completion":                 "bg:#212121 #f5f5f5",
                "path-completion":                    "bg:#800080 #f5f5f5",
                "file-completion":                    "bg:#9040b2 #f5f5f5",
                "history-completion-selected":        "bg:#efefef #b3d7ff",
                "path-completion-selected":           "bg:#efefef #b3d7ff",
                "file-completion-selected":           "bg:#efefef #b3d7ff"
            },
            "map-nt-cmd":                             True,
            "show-splash-screen":                     True,
            "plugin_prefix":                          None
        })

    # CREATE A SERVER CONFIGURATION FILE IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    _symserver_config_path_ = config_manager.get_config_path('symserver.config.json')
    if not os.path.exists(_symserver_config_path_):
        config_manager.save_config('symserver.config.json', {})

    # Get appropriate config path (debug mode handling is now in config_manager)
    _symai_config_path_ = config_manager.get_config_path('symai.config.json')

    if not os.path.exists(_symai_config_path_):
        setup_wizard(_symai_config_path_, show_wizard=False)
        CustomUserWarning(f'No configuration file found for the environment. A new configuration file has been created at {_symai_config_path_}. Please configure your environment.')
        sys.exit(1)

    # Load and manage configurations
    _symai_config_ = config_manager.load_config('symai.config.json')
    _tmp_symai_config_ = _symai_config_.copy()

    # MIGRATE THE ENVIRONMENT VARIABLES
    # *==========================================================================================================*
    if 'SPEECH_ENGINE_MODEL' in _symai_config_:
        updates = {
            'SPEECH_TO_TEXT_ENGINE_MODEL': _symai_config_['SPEECH_ENGINE_MODEL'],
            'TEXT_TO_SPEECH_ENGINE_MODEL': "tts-1"
        }
        config_manager.migrate_config('symai.config.json', updates)
        del _symai_config_['SPEECH_ENGINE_MODEL']
        del _tmp_symai_config_['SPEECH_ENGINE_MODEL']

    if 'COLLECTION_URI' not in _symai_config_:
        updates = {
            'COLLECTION_URI': "mongodb+srv://User:vt3epocXitd6WlQ6@extensityai.c1ajxxy.mongodb.net/?retryWrites=true&w=majority",
            'COLLECTION_DB': "ExtensityAI",
            'COLLECTION_STORAGE': "SymbolicAI",
            'SUPPORT_COMMUNITY': False
        }
        config_manager.migrate_config('symai.config.json', updates)
        with ConsoleStyle('info') as console:
            msg = 'Currently you are sharing your user experience with us by uploading the data to our research server, and thereby helping us improve future models and the overall SymbolicAI experience. We thank you very much for supporting the research community! If you wish to disable the data collection option go to your .symai config situated in your home directory or set the environment variable `SUPPORT_COMMUNITY` to `False`.'
            console.print(msg)

    # POST-MIGRATION CHECKS
    # *==============================================================================================================*
    if 'TEXT_TO_SPEECH_ENGINE_API_KEY' not in _symai_config_:
        updates = {
            'TEXT_TO_SPEECH_ENGINE_API_KEY': _symai_config_.get('NEUROSYMBOLIC_ENGINE_API_KEY', '')
        }
        config_manager.migrate_config('symai.config.json', updates)

    # Load all configurations
    _symai_config_ = config_manager.load_config('symai.config.json')
    _symsh_config_ = config_manager.load_config('symsh.config.json')
    _symserver_config_ = config_manager.load_config('symserver.config.json')

    # MIGRATE THE SHELL SPLASH SCREEN CONFIGURATION
    # *==============================================================================================================*
    if 'show-splash-screen' not in _symsh_config_:
        config_manager.migrate_config('symsh.config.json', {'show-splash-screen': True})

    # CHECK IF THE USER HAS A NEUROSYMBOLIC API KEY
    # *==============================================================================================================*
    if not (
        _symai_config_['NEUROSYMBOLIC_ENGINE_MODEL'].lower().startswith('llama') or \
        _symai_config_['NEUROSYMBOLIC_ENGINE_MODEL'].lower().startswith('huggingface')) \
        and \
            (
            _symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or \
            len(_symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']) == 0):
            # Try to fallback to the global (home) config if environment is not home
            if config_manager.config_dir != config_manager._home_config_dir:
                CustomUserWarning(f"You didn't configure your environment ({config_manager.config_dir})! Falling back to the global ({config_manager._home_config_dir}) configuration if it exists.")
                # Force loading from home
                _symai_config_ = config_manager.load_config('symai.config.json', fallback_to_home=True)
                _symsh_config_ = config_manager.load_config('symsh.config.json', fallback_to_home=True)
                _symserver_config_ = config_manager.load_config('symserver.config.json', fallback_to_home=True)

            # If still not valid, warn and exit
            if not _symai_config_.get('NEUROSYMBOLIC_ENGINE_API_KEY'):
                CustomUserWarning('The mandatory neuro-symbolic engine is not initialized. Please set NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY.')
                sys.exit(1)

    settings.SYMAI_CONFIG = _symai_config_
    settings.SYMSH_CONFIG = _symsh_config_
    settings.SYMSERVER_CONFIG = _symserver_config_

def run_setup_wizard(file_path = None):
    if file_path is None:
        file_path = config_manager.get_config_path('symai.config.json')
    setup_wizard(file_path)


def run_server():
    _symserver_config_ = {}
    if settings.SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL").startswith("llama") or settings.SYMAI_CONFIG.get("EMBEDDING_ENGINE_MODEL").startswith("llama"):
        from .server.llama_cpp_server import llama_cpp_server

        command, args = llama_cpp_server()
        _symserver_config_.update(zip(args[::2], args[1::2]))
        _symserver_config_['online'] = True

        config_manager.save_config("symserver.config.json", _symserver_config_)

        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            print("Server stopped!")
        except Exception as e:
            print(f"Error running server: {e}")
        finally:
            config_manager.save_config("symserver.config.json", {'online': False})

    elif settings.SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL").startswith("huggingface"):
        from .server.huggingface_server import huggingface_server

        command, args = huggingface_server()
        _symserver_config_.update(vars(args))
        _symserver_config_['online'] = True

        config_manager.save_config("symserver.config.json", _symserver_config_)

        try:
            command(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("Server stopped!")
        except Exception as e:
            print(f"Error running server: {e}")
        finally:
            config_manager.save_config("symserver.config.json", {'online': False})
    else:
        raise CustomUserWarning("You're trying to run a local server without a valid neuro-symbolic engine model. Please set a valid model in your configuration file. Current available options are 'llamacpp' and 'huggingface'.")


# *==============================================================================================================*
def format_config_content(config: dict) -> str:
    """Format config content for display, truncating API keys."""
    formatted = {}
    for k, v in config.items():
        if isinstance(v, str) and ('KEY' in k or 'URI' in k) and v:
            # Show first/last 4 chars of keys/URIs
            formatted[k] = f"{v[:4]}...{v[-4:]}" if len(v) > 8 else v
        else:
            formatted[k] = v
    return json.dumps(formatted, indent=2)

def display_config():
    """Display all configuration paths and their content."""

    console = Console()

    # Create header
    console.print(Panel.fit(
        f"[bold cyan]SymbolicAI Configuration Inspector v{__version__}[/bold cyan]",
        border_style="cyan"
    ))

    # Create main tree
    tree = Tree("[bold]Configuration Locations[/bold]")

    # Debug config
    debug_branch = tree.add("[yellow]Debug Mode Config (CWD)[/yellow]")
    debug_config = config_manager._debug_dir / 'symai.config.json'
    if debug_config.exists():
        with open(debug_config) as f:
            content = json.load(f)
        debug_branch.add(f"📄 [green]{debug_config}[/green]\n{format_config_content(content)}")
    else:
        debug_branch.add("[dim]No debug config found[/dim]")

    # Environment config
    env_branch = tree.add("[yellow]Environment Config[/yellow]")
    env_configs = {
        'symai.config.json': '⚙️',
        'symsh.config.json': '🖥️',
        'symserver.config.json': '🌐'
    }

    for config_file, icon in env_configs.items():
        config_path = config_manager._env_config_dir / config_file
        if config_path.exists():
            with open(config_path) as f:
                content = json.load(f)
            env_branch.add(f"{icon} [green]{config_path}[/green]\n{format_config_content(content)}")
        else:
            env_branch.add(f"[dim]{icon} {config_file} (not found)[/dim]")

    # Home (global) config
    home_branch = tree.add("[yellow]Home Directory Config (Global)[/yellow]")
    for config_file, icon in env_configs.items():
        config_path = config_manager._home_config_dir / config_file
        if config_path.exists():
            with open(config_path) as f:
                content = json.load(f)
            home_branch.add(f"{icon} [green]{config_path}[/green]\n{format_config_content(content)}")
        else:
            home_branch.add(f"[dim]{icon} {config_file} (not found)[/dim]")

    # Active configuration summary
    summary = Table(show_header=True, header_style="bold magenta")
    summary.add_column("Configuration Type")
    summary.add_column("Active Path")

    active_paths = {
        "Primary Config Dir": config_manager.config_dir,
        "symai.config.json": config_manager.get_config_path('symai.config.json'),
        "symsh.config.json": config_manager.get_config_path('symsh.config.json'),
        "symserver.config.json": config_manager.get_config_path('symserver.config.json')
    }

    for config_type, path in active_paths.items():
        summary.add_row(config_type, str(path))

    # Print everything
    console.print(tree)
    console.print("\n[bold]Active Configuration Summary:[/bold]")
    console.print(summary)

    # Print help
    console.print("\n[bold]Legend:[/bold]")
    console.print("⚙️  symai.config.json (Main SymbolicAI configuration)")
    console.print("🖥️  symsh.config.json (Shell configuration)")
    console.print("🌐  symserver.config.json (Server configuration)")
    console.print("\n[dim]Note: API keys and URIs are truncated for security[/dim]")
# *==============================================================================================================*


def setup_wizard(_symai_config_path_, show_wizard=True):
    _user_config_                   = show_menu(show_wizard=show_wizard)
    _nesy_engine_api_key            = _user_config_['nesy_engine_api_key']
    _nesy_engine_model              = _user_config_['nesy_engine_model']
    _symbolic_engine_api_key        = _user_config_['symbolic_engine_api_key']
    _symbolic_engine_model          = _user_config_['symbolic_engine_model']
    _embedding_engine_api_key       = _user_config_['embedding_engine_api_key']
    _embedding_model                = _user_config_['embedding_model']
    _drawing_engine_api_key         = _user_config_['drawing_engine_api_key']
    _drawing_engine_model           = _user_config_['drawing_engine_model']
    _vision_engine_model            = _user_config_['vision_engine_model']
    _search_engine_api_key          = _user_config_['search_engine_api_key']
    _search_engine_model            = _user_config_['search_engine_model']
    _ocr_engine_api_key             = _user_config_['ocr_engine_api_key']
    _speech_to_text_engine_model    = _user_config_['speech_to_text_engine_model']
    _text_to_speech_engine_api_key  = _user_config_['text_to_speech_engine_api_key']
    _text_to_speech_engine_model    = _user_config_['text_to_speech_engine_model']
    _text_to_speech_engine_voice    = _user_config_['text_to_speech_engine_voice']
    _indexing_engine_api_key        = _user_config_['indexing_engine_api_key']
    _indexing_engine_environment    = _user_config_['indexing_engine_environment']
    _caption_engine_environment     = _user_config_['caption_engine_environment']
    _support_comminity              = _user_config_['support_community']

    config_manager.save_config(_symai_config_path_, {
        "NEUROSYMBOLIC_ENGINE_API_KEY":   _nesy_engine_api_key,
        "NEUROSYMBOLIC_ENGINE_MODEL":     _nesy_engine_model,
        "SYMBOLIC_ENGINE_API_KEY":        _symbolic_engine_api_key,
        "SYMBOLIC_ENGINE":                _symbolic_engine_model,
        "EMBEDDING_ENGINE_API_KEY":       _embedding_engine_api_key,
        "EMBEDDING_ENGINE_MODEL":         _embedding_model,
        "DRAWING_ENGINE_API_KEY":         _drawing_engine_api_key,
        "DRAWING_ENGINE_MODEL":           _drawing_engine_model,
        "VISION_ENGINE_MODEL":            _vision_engine_model,
        "SEARCH_ENGINE_API_KEY":          _search_engine_api_key,
        "SEARCH_ENGINE_MODEL":            _search_engine_model,
        "OCR_ENGINE_API_KEY":             _ocr_engine_api_key,
        "SPEECH_TO_TEXT_ENGINE_MODEL":    _speech_to_text_engine_model,
        "TEXT_TO_SPEECH_ENGINE_API_KEY":  _text_to_speech_engine_api_key,
        "TEXT_TO_SPEECH_ENGINE_MODEL":    _text_to_speech_engine_model,
        "TEXT_TO_SPEECH_ENGINE_VOICE":    _text_to_speech_engine_voice,
        "INDEXING_ENGINE_API_KEY":        _indexing_engine_api_key,
        "INDEXING_ENGINE_ENVIRONMENT":    _indexing_engine_environment,
        "CAPTION_ENGINE_MODEL":           _caption_engine_environment,
        "COLLECTION_URI":                 "mongodb+srv://User:vt3epocXitd6WlQ6@extensityai.c1ajxxy.mongodb.net/?retryWrites=true&w=majority",
        "COLLECTION_DB":                  "ExtensityAI",
        "COLLECTION_STORAGE":             "SymbolicAI",
        "SUPPORT_COMMUNITY":              _support_comminity
    })


_start_symai()

from .backend.base import Engine
from .components import Function, PrimitiveDisabler
from .core import few_shot, zero_shot
from .extended import Conversation
from .functional import EngineRepository
from .imports import Import
from .interfaces import Interface
from .post_processors import PostProcessor
from .pre_processors import PreProcessor
from .prompts import Prompt, PromptLanguage, PromptRegistry
from .shell import Shell
from .strategy import Strategy
from .symbol import Call, Expression, GlobalSymbolPrimitive, Metadata, Symbol
