import argparse
import subprocess
import sys
from pathlib import Path

from symai.utils import UserMessage


def vllm_server():
    """Build the subprocess command for vLLM's OpenAI-compatible server.

    Returns (command, args). The outer `run_server` is what actually executes
    the command via `subprocess.run`. Everything after the wrapper-owned flags
    is forwarded to vLLM untouched, so the full `vllm serve` CLI surface
    (--model, --host, --port, --tensor-parallel-size, --dtype, --max-model-len,
    --gpu-memory-utilization, --enable-auto-tool-choice, --tool-call-parser,
    --served-model-name, ...) is available.

    Two entrypoint models, mirroring `llama_cpp_server.py`:

    * Default: use the current environment — `python -m vllm.entrypoints.openai.api_server`
      (`--entrypoint module`) or `vllm serve` (`--entrypoint cli`). Requires vLLM
      importable from `sys.executable` or `vllm` on `PATH`.
    * `--vllm-python-path /path/to/venv/bin/python`: run vLLM via that interpreter
      instead. Use this on macOS where vLLM usually lives in its own source-built
      venv outside the symai project (see docs/source/ENGINES/local_engine.md).
    """
    parser = argparse.ArgumentParser(description="A wrapper for vLLM server.", add_help=False)
    parser.add_argument(
        "--help", action="store_true", help="Show available options for vLLM server."
    )
    parser.add_argument(
        "--entrypoint",
        choices=["module", "cli"],
        default="module",
        help=(
            "Choose vLLM entrypoint: 'module' -> `python -m vllm.entrypoints.openai.api_server`; "
            "'cli' -> `vllm serve` (requires `vllm` on PATH). Ignored when "
            "--vllm-python-path is set."
        ),
    )
    parser.add_argument(
        "--vllm-python-path",
        type=str,
        default=None,
        help=(
            "Path to a Python interpreter with vLLM installed (e.g. a source-built "
            "venv on macOS). When set, vLLM is launched via "
            "`<path> -m vllm.entrypoints.openai.api_server ...` and --entrypoint is ignored."
        ),
    )

    main_args, vllm_args = parser.parse_known_args()

    if main_args.vllm_python_path:
        python = Path(main_args.vllm_python_path)
        if not python.exists():
            UserMessage(
                f"Python interpreter not found at {python}", raise_with=SystemExit
            )
        base = [str(python), "-m", "vllm.entrypoints.openai.api_server"]
    elif main_args.entrypoint == "module":
        base = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    else:
        base = ["vllm", "serve"]

    if main_args.help:
        subprocess.run([*base, "--help"], check=False)
        sys.exit(0)

    command = [*base, *vllm_args]
    return command, vllm_args
