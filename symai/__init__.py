import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import warnings

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .backend import settings
from .utils import UserMessage

# do not remove - hides the libraries' debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface").setLevel(logging.ERROR)


warnings.simplefilter("ignore")

# set the environment variable for the transformers library
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create singleton instance
config_manager = settings.SymAIConfig()

SYMAI_VERSION = "1.15.0"
__version__ = SYMAI_VERSION
__root_dir__ = config_manager.config_dir


def _start_symai():
    # Create config directories if they don't exist
    config_manager._env_config_dir.mkdir(parents=True, exist_ok=True)
    config_manager._home_config_dir.mkdir(parents=True, exist_ok=True)

    # CREATE THE SHELL CONFIGURATION FILE IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    _symsh_config_path_ = config_manager.get_config_path("symsh.config.json")
    if not _symsh_config_path_.exists():
        config_manager.save_config(
            "symsh.config.json",
            {
                "colors": {
                    "completion-menu.completion.current": "bg:#323232 #212121",
                    "completion-menu.completion": "bg:#800080 #212121",
                    "scrollbar.background": "bg:#222222",
                    "scrollbar.button": "bg:#776677",
                    "history-completion": "bg:#212121 #f5f5f5",
                    "path-completion": "bg:#800080 #f5f5f5",
                    "file-completion": "bg:#9040b2 #f5f5f5",
                    "history-completion-selected": "bg:#efefef #b3d7ff",
                    "path-completion-selected": "bg:#efefef #b3d7ff",
                    "file-completion-selected": "bg:#efefef #b3d7ff",
                },
                "map-nt-cmd": True,
                "show-splash-screen": True,
                "plugin_prefix": None,
            },
        )

    # CREATE A SERVER CONFIGURATION FILE IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    _symserver_config_path_ = config_manager.get_config_path("symserver.config.json")
    if not _symserver_config_path_.exists():
        config_manager.save_config("symserver.config.json", {})

    # Get appropriate config path (debug mode handling is now in config_manager)
    _symai_config_path_ = config_manager.get_config_path("symai.config.json")

    if not _symai_config_path_.exists():
        setup_wizard(_symai_config_path_)
        UserMessage(
            f"No configuration file found for the environment. A new configuration file has been created at {_symai_config_path_}. Please configure your environment."
        )
        sys.exit(1)

    # Load and manage configurations
    symai_config = config_manager.load_config("symai.config.json")

    # POST-MIGRATION CHECKS
    # *==============================================================================================================*
    if "TEXT_TO_SPEECH_ENGINE_API_KEY" not in symai_config:
        updates = {
            "TEXT_TO_SPEECH_ENGINE_API_KEY": symai_config.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
        }
        config_manager.migrate_config("symai.config.json", updates)

    # Load all configurations
    symai_config = config_manager.load_config("symai.config.json")
    symsh_config = config_manager.load_config("symsh.config.json")
    symserver_config = config_manager.load_config("symserver.config.json")

    # CHECK IF THE USER HAS A NEUROSYMBOLIC API KEY
    # *==============================================================================================================*
    if not (
        symai_config["NEUROSYMBOLIC_ENGINE_MODEL"].lower().startswith("llama")
        or symai_config["NEUROSYMBOLIC_ENGINE_MODEL"].lower().startswith("vllm")
        or symai_config["NEUROSYMBOLIC_ENGINE_MODEL"].lower().startswith("huggingface")
    ) and (
        symai_config["NEUROSYMBOLIC_ENGINE_API_KEY"] is None
        or len(symai_config["NEUROSYMBOLIC_ENGINE_API_KEY"]) == 0
    ):
        # Try to fallback to the global (home) config if environment is not home
        if config_manager.config_dir != config_manager._home_config_dir:
            UserMessage(
                f"You didn't configure your environment ({config_manager.config_dir})! Falling back to the global ({config_manager._home_config_dir}) configuration if it exists."
            )
            # Force loading from home
            symai_config = config_manager.load_config("symai.config.json", fallback_to_home=True)
            symsh_config = config_manager.load_config("symsh.config.json", fallback_to_home=True)
            symserver_config = config_manager.load_config(
                "symserver.config.json", fallback_to_home=True
            )

        # If still not valid, warn and exit
        if not symai_config.get("NEUROSYMBOLIC_ENGINE_API_KEY"):
            UserMessage(
                "The mandatory neuro-symbolic engine is not initialized. Please set NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            sys.exit(1)

    settings.SYMAI_CONFIG = symai_config
    settings.SYMSH_CONFIG = symsh_config
    settings.SYMSERVER_CONFIG = symserver_config
    return symai_config, symsh_config, symserver_config


from .server.qdrant_server import _QDRANT_SERVER_FLAGS  # noqa: E402


def run_server():
    _symserver_config_ = {}

    def _save_symserver_config(config: dict, *, include_home: bool = False) -> None:
        config_manager.save_config("symserver.config.json", config)
        if include_home:
            config_manager.save_config("symserver.config.json", config, fallback_to_home=True)

    # Check for explicit Qdrant server request via command line.
    # Matches either a literal "qdrant" substring in any arg (e.g. --docker-image qdrant/qdrant)
    # or any flag that is exclusive to qdrant_server.py (e.g. --docker-detach, --max-workers).
    qdrant_requested = any(
        "qdrant" in arg.lower() or arg in _QDRANT_SERVER_FLAGS for arg in sys.argv[1:]
    )
    rag_requested = any(arg == "--rag" or arg.startswith("--rag-api") for arg in sys.argv[1:])
    lean4_requested = "--lean4" in sys.argv[1:]
    uvicorn_reload_default = os.getenv("UVICORN_RELOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _args_to_config(args: list[str]) -> dict[str, object]:
        """
        Convert a CLI argv-like list into a dict.

        Supports:
        - key/value pairs: ["--port", "6333"]
        - boolean flags: ["--no-cache"]
        """
        cfg = {}
        i = 0
        while i < len(args):
            tok = args[i]
            if not tok.startswith("--"):
                i += 1
                continue
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                cfg[tok] = args[i + 1]
                i += 2
            else:
                cfg[tok] = True
                i += 1
        return cfg

    def _wait_for_qdrant(
        url: str,
        *,
        api_key: str | None = None,
        max_retries: int = 40,
        delay_s: float = 0.5,
    ) -> bool:
        max_retries = int(os.getenv("QDRANT_WAIT_RETRIES", str(max_retries)))
        delay_s = float(os.getenv("QDRANT_WAIT_DELAY", str(delay_s)))
        headers = {}
        if api_key:
            headers["api-key"] = api_key
        for _ in range(max_retries):
            try:
                req = urllib.request.Request(f"{url}/collections", headers=headers)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if 200 <= resp.status < 500:
                        return True
            except Exception:
                time.sleep(delay_s)
        return False

    if (
        qdrant_requested
        or settings.SYMAI_CONFIG.get("INDEXING_ENGINE") == "qdrant"
        or any(
            "qdrant" in str(v).lower()
            for v in [
                settings.SYMAI_CONFIG.get("INDEXING_ENGINE_API_KEY", ""),
                settings.SYMAI_CONFIG.get("INDEXING_ENGINE_URL", ""),
            ]
        )
    ):
        from .server.qdrant_server import qdrant_server  # noqa

        # Optional RAG API companion server (FastAPI/uvicorn) configuration.
        # We parse these args first, then pass the remaining args to the Qdrant wrapper.
        rag_parser = argparse.ArgumentParser(add_help=False)
        rag_parser.add_argument(
            "--rag", "--rag-api", dest="rag_api", action="store_true", default=False
        )
        rag_parser.add_argument(
            "--rag-host",
            "--rag-api-host",
            dest="rag_api_host",
            type=str,
            default=os.getenv("RAG_API_HOST", "0.0.0.0"),
        )
        rag_parser.add_argument(
            "--rag-port",
            "--rag-api-port",
            dest="rag_api_port",
            type=int,
            default=int(os.getenv("RAG_API_PORT", "8080")),
        )
        rag_parser.add_argument(
            "--rag-workers",
            "--rag-api-workers",
            dest="rag_api_workers",
            type=int,
            default=int(os.getenv("RAG_API_WORKERS", "1")),
        )
        rag_parser.add_argument(
            "--rag-token",
            "--rag-api-token",
            dest="rag_api_token",
            type=str,
            default=os.getenv("RAG_API_TOKEN", ""),
        )
        rag_parser.add_argument(
            "--rag-reload",
            "--rag-api-reload",
            dest="rag_api_reload",
            action="store_true",
            default=uvicorn_reload_default,
            help="Enable uvicorn reload (dev only).",
        )

        rag_args, qdrant_argv = rag_parser.parse_known_args(sys.argv[1:])
        rag_enabled = bool(rag_args.rag_api or rag_requested)

        command, args = qdrant_server(qdrant_argv)
        qdrant_cfg = _args_to_config(args)
        _symserver_config_.update(qdrant_cfg)
        _symserver_config_["online"] = True
        qdrant_port = int(str(qdrant_cfg.get("--port", "6333")))
        qdrant_url = f"http://127.0.0.1:{qdrant_port}"
        _symserver_config_["url"] = qdrant_url

        if rag_enabled:
            _symserver_config_["rag_api"] = {
                "enabled": True,
                "host": rag_args.rag_api_host,
                "port": rag_args.rag_api_port,
                "workers": rag_args.rag_api_workers,
                "reload": bool(rag_args.rag_api_reload),
                "token_required": bool(rag_args.rag_api_token),
                "url": f"http://{rag_args.rag_api_host}:{rag_args.rag_api_port}",
            }

        _save_symserver_config(_symserver_config_, include_home=True)

        qdrant_proc = None
        api_proc = None
        qdrant_detached = False
        try:
            if not rag_enabled:
                subprocess.run(command, check=True)
                return

            qdrant_proc = subprocess.Popen(command)
            qdrant_detached = "-d" in command
            if qdrant_detached:
                qdrant_exit = qdrant_proc.wait(timeout=30)
                if qdrant_exit != 0:
                    msg = f"Qdrant docker process exited with code {qdrant_exit}"
                    raise RuntimeError(msg)

            qdrant_api_key = str(qdrant_cfg.get("--api-key", "")).strip() or None
            if not _wait_for_qdrant(qdrant_url, api_key=qdrant_api_key):
                msg = "Qdrant did not become ready in time"
                raise RuntimeError(msg)

            uvicorn_cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "symai.server.qdrant_rag_api:app",
                "--host",
                rag_args.rag_api_host,
                "--port",
                str(rag_args.rag_api_port),
                "--workers",
                str(rag_args.rag_api_workers),
            ]
            if rag_args.rag_api_reload:
                uvicorn_cmd.append("--reload")

            api_env = os.environ.copy()
            api_env["SYMAI_QDRANT_URL"] = qdrant_url
            api_env["INDEXING_ENGINE_URL"] = qdrant_url
            # Keep RAG API->Qdrant auth aligned with how qdrant was started.
            # Priority: explicit INDEXING_ENGINE_API_KEY > qdrant --api-key.
            indexing_api_key = (
                str(api_env.get("INDEXING_ENGINE_API_KEY", "")).strip() or qdrant_api_key or ""
            )
            if indexing_api_key:
                api_env["INDEXING_ENGINE_API_KEY"] = indexing_api_key
            if rag_args.rag_api_token:
                api_env["RAG_API_TOKEN"] = rag_args.rag_api_token

            api_proc = subprocess.Popen(uvicorn_cmd, env=api_env)

            # Wait until one of the processes exits.
            while True:
                api_rc = api_proc.poll()
                if api_rc is not None:
                    msg = f"RAG API exited with code {api_rc}"
                    raise RuntimeError(msg)
                if not qdrant_detached:
                    q_rc = qdrant_proc.poll()
                    if q_rc is not None:
                        msg = f"Qdrant exited with code {q_rc}"
                        raise RuntimeError(msg)
                time.sleep(0.25)
        except KeyboardInterrupt:
            UserMessage("Server stopped.", style="success")
        except Exception as e:
            UserMessage(f"Error running server: {e}")
        finally:
            # Best-effort shutdown for companion processes (if used).
            try:
                if rag_enabled and api_proc is not None:
                    api_proc.terminate()
            except Exception:
                pass
            try:
                if rag_enabled and qdrant_proc is not None and not qdrant_detached:
                    qdrant_proc.terminate()
            except Exception:
                pass

            # If Qdrant was started detached in docker mode, stop the container explicitly.
            try:
                docker_name = _symserver_config_.get("--docker-container-name")
                if rag_enabled and qdrant_detached and docker_name:
                    subprocess.run(["docker", "stop", str(docker_name)], check=False)
            except Exception:
                pass

            _symserver_config_["online"] = False
            _save_symserver_config(_symserver_config_)
    # TODO: support NEUROSYMBOLIC_ENGINE_URL / EMBEDDING_ENGINE_URL in SYMAI_CONFIG
    # so users can point to an externally-hosted server (e.g. vLLM on another host,
    # llama.cpp already running) without requiring symserver to manage the subprocess.
    elif settings.SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL").startswith(
        "llama"
    ) or settings.SYMAI_CONFIG.get("EMBEDDING_ENGINE_MODEL").startswith("llama"):
        # Keep optional llama_cpp dependencies lazy.
        from .server.llama_cpp_server import llama_cpp_server  # noqa

        command, args = llama_cpp_server()
        _symserver_config_.update(_args_to_config(args))
        _symserver_config_["online"] = True

        # @NOTE: Save in both places since you can start the server from anywhere and still not have a nesy engine configured
        _save_symserver_config(_symserver_config_, include_home=True)

        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            UserMessage("Server stopped.", style="success")
        except Exception as e:
            UserMessage(f"Error running server: {e}")
        finally:
            _symserver_config_["online"] = False
            _save_symserver_config(_symserver_config_)

    elif settings.SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL").startswith("vllm"):
        # Keep optional vllm dependencies lazy.
        from .server.vllm_server import vllm_server  # noqa

        command, args = vllm_server()
        # NOTE: _args_to_config handles bare boolean flags (e.g. --enable-auto-tool-choice,
        # --enforce-eager) correctly; naive zip(args[::2], args[1::2]) misaligns when a
        # boolean flag is followed by another flag.
        _symserver_config_.update(_args_to_config(args))
        _symserver_config_["online"] = True

        _save_symserver_config(_symserver_config_, include_home=True)

        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            UserMessage("Server stopped.", style="success")
        except Exception as e:
            UserMessage(f"Error running server: {e}")
        finally:
            _symserver_config_["online"] = False
            _save_symserver_config(_symserver_config_)

    elif settings.SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL").startswith("huggingface"):
        # HuggingFace server stack is optional; import only when requested.
        from .server.huggingface_server import huggingface_server  # noqa

        command, args = huggingface_server()
        _symserver_config_.update(vars(args))
        _symserver_config_["online"] = True

        _save_symserver_config(_symserver_config_)

        try:
            command(host=args.host, port=args.port)
        except KeyboardInterrupt:
            UserMessage("Server stopped.", style="success")
        except Exception as e:
            UserMessage(f"Error running server: {e}")
        finally:
            _symserver_config_["online"] = False
            _save_symserver_config(_symserver_config_)

    elif lean4_requested:
        # Lean4 server - check FORMAL_ENGINE and start container + FastAPI
        from .server.lean4_server import lean4_server  # noqa

        command, args = lean4_server()
        _symserver_config_.update(_args_to_config(args))
        _symserver_config_["online"] = True

        # Extract port from command (uvicorn --port X)
        port = 8000
        for i, arg in enumerate(command):
            if arg == "--port" and i + 1 < len(command):
                port = int(command[i + 1])
                break
        _symserver_config_["url"] = f"http://localhost:{port}"

        _save_symserver_config(_symserver_config_, include_home=True)

        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            UserMessage("Server stopped.", style="success")
        except Exception as e:
            UserMessage(f"Error running server: {e}")
        finally:
            _symserver_config_["online"] = False
            _save_symserver_config(_symserver_config_)

    else:
        msg = (
            "You're trying to run a local server without a recognised engine configuration. "
            "Options:\n"
            "  - Qdrant (indexing/RAG):         set INDEXING_ENGINE=qdrant in symai.config.json, "
            "or pass any qdrant_server flag (e.g. symserver --docker-detach)\n"
            "  - llama.cpp (neuro-symbolic):    set NEUROSYMBOLIC_ENGINE_MODEL=llamacpp or "
            "EMBEDDING_ENGINE_MODEL=llamacpp in symai.config.json\n"
            "  - vLLM (neuro-symbolic):         set NEUROSYMBOLIC_ENGINE_MODEL=vllm in symai.config.json\n"
            "  - HuggingFace (neuro-symbolic):  set NEUROSYMBOLIC_ENGINE_MODEL=huggingface "
            "in symai.config.json\n"
            "  - Lean4 (formal):                symserver --lean4 (requires FORMAL_ENGINE=local)"
        )
        UserMessage(msg, raise_with=ValueError)


# *==============================================================================================================*
def format_config_content(config: dict) -> str:
    """Format config content for display, truncating API keys."""
    formatted = {}
    for k, v in config.items():
        if isinstance(v, str) and ("KEY" in k or "URI" in k) and v:
            # Show first/last 4 chars of keys/URIs
            formatted[k] = f"{v[:4]}...{v[-4:]}" if len(v) > 8 else v
        else:
            formatted[k] = v
    return json.dumps(formatted, indent=2)


def display_config():
    """Display all configuration paths and their content."""

    console = Console()

    # Create header
    console.print(
        Panel.fit(
            f"[bold cyan]SymbolicAI Configuration Inspector v{__version__}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Create main tree
    tree = Tree("[bold]Configuration Locations[/bold]")

    # Debug config
    debug_branch = tree.add("[yellow]Debug Mode Config (CWD)[/yellow]")
    debug_config = config_manager._debug_dir / "symai.config.json"
    if debug_config.exists():
        with debug_config.open() as f:
            content = json.load(f)
        debug_branch.add(f"📄 [green]{debug_config}[/green]\n{format_config_content(content)}")
    else:
        debug_branch.add("[dim]No debug config found[/dim]")

    # Environment config
    env_branch = tree.add("[yellow]Environment Config[/yellow]")
    env_configs = {
        "symai.config.json": "⚙️",
        "symsh.config.json": "🖥️",
        "symserver.config.json": "🌐",
    }

    for config_file, icon in env_configs.items():
        config_path = config_manager._env_config_dir / config_file
        if config_path.exists():
            with config_path.open() as f:
                content = json.load(f)
            env_branch.add(f"{icon} [green]{config_path}[/green]\n{format_config_content(content)}")
        else:
            env_branch.add(f"[dim]{icon} {config_file} (not found)[/dim]")

    # Home (global) config
    home_branch = tree.add("[yellow]Home Directory Config (Global)[/yellow]")
    for config_file, icon in env_configs.items():
        config_path = config_manager._home_config_dir / config_file
        if config_path.exists():
            with config_path.open() as f:
                content = json.load(f)
            home_branch.add(
                f"{icon} [green]{config_path}[/green]\n{format_config_content(content)}"
            )
        else:
            home_branch.add(f"[dim]{icon} {config_file} (not found)[/dim]")

    # Active configuration summary
    summary = Table(show_header=True, header_style="bold magenta")
    summary.add_column("Configuration Type")
    summary.add_column("Active Path")

    active_paths = {
        "Primary Config Dir": config_manager.get_active_config_dir(),
        "symai.config.json": config_manager.get_active_path("symai.config.json"),
        "symsh.config.json": config_manager.get_active_path("symsh.config.json"),
        "symserver.config.json": config_manager.get_active_path("symserver.config.json"),
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


def setup_wizard(_symai_config_path_):
    config_manager.save_config(
        _symai_config_path_,
        {
            "NEUROSYMBOLIC_ENGINE_API_KEY": "",
            "NEUROSYMBOLIC_ENGINE_MODEL": "",
            "SYMBOLIC_ENGINE_API_KEY": "",
            "SYMBOLIC_ENGINE": "",
            "FORMAL_ENGINE_API_KEY": "",
            "FORMAL_ENGINE": "",
            "EMBEDDING_ENGINE_API_KEY": "",
            "EMBEDDING_ENGINE_MODEL": "",
            "DRAWING_ENGINE_API_KEY": "",
            "DRAWING_ENGINE_MODEL": "",
            "VISION_ENGINE_MODEL": "",
            "SEARCH_ENGINE_API_KEY": "",
            "SEARCH_ENGINE_MODEL": "",
            "OCR_ENGINE_API_KEY": "",
            "OCR_ENGINE_MODEL": "",
            "SPEECH_TO_TEXT_ENGINE_MODEL": "",
            "SPEECH_TO_TEXT_API_KEY": "",
            "TEXT_TO_SPEECH_ENGINE_API_KEY": "",
            "TEXT_TO_SPEECH_ENGINE_MODEL": "",
            "TEXT_TO_SPEECH_ENGINE_VOICE": "",
            "INDEXING_ENGINE_API_KEY": "",
            "INDEXING_ENGINE_ENVIRONMENT": "",
            "CAPTION_ENGINE_MODEL": "",
        },
    )


_symai_config_, _symsh_config_, _symserver_config_ = _start_symai()

from .backend.base import Engine  # noqa
from .components import Function, PrimitiveDisabler  # noqa
from .core import few_shot, zero_shot  # noqa
from .extended import Conversation  # noqa
from .functional import EngineRepository  # noqa
from .imports import Import  # noqa
from .interfaces import Interface  # noqa
from .post_processors import PostProcessor  # noqa
from .pre_processors import PreProcessor  # noqa
from .prompts import Prompt, PromptLanguage, PromptRegistry  # noqa
from .shell import Shell  # noqa
from .strategy import Strategy  # noqa
from .symbol import Call, Expression, GlobalSymbolPrimitive, Metadata, Symbol  # noqa

__all__ = [
    "SYMAI_VERSION",
    "Call",
    "Conversation",
    "Engine",
    "EngineRepository",
    "Expression",
    "Function",
    "GlobalSymbolPrimitive",
    "Import",
    "Interface",
    "Metadata",
    "PostProcessor",
    "PreProcessor",
    "PrimitiveDisabler",
    "Prompt",
    "PromptLanguage",
    "PromptRegistry",
    "Shell",
    "Strategy",
    "Symbol",
    "__root_dir__",
    "__version__",
    "config_manager",
    "few_shot",
    "run_server",
    "setup_wizard",
    "zero_shot",
]
