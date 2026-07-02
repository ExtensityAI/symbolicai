import argparse
import subprocess
from pathlib import Path


def llama_cpp_server() -> tuple:
    parser = argparse.ArgumentParser(
        description="A wrapper for the llama.cpp server binary.", add_help=False
    )
    parser.add_argument(
        "--help", action="store_true", help="Show available options for llama.cpp server."
    )
    parser.add_argument("--cpp-server-path", type=str, help="Path to llama.cpp server executable")

    main_args, llama_cpp_args = parser.parse_known_args()

    if not main_args.cpp_server_path:
        if main_args.help:
            parser.print_help()
            raise SystemExit(0)
        msg = "Error: --cpp-server-path is required for the llama.cpp server binary"
        raise SystemExit(msg)

    cpp_server_path = Path(main_args.cpp_server_path)
    if not cpp_server_path.exists():
        msg = f"Error: Executable not found at {cpp_server_path}"
        raise SystemExit(msg)

    if main_args.help:
        subprocess.run([cpp_server_path, "--help"], check=False)
        raise SystemExit(0)

    command = [cpp_server_path, *llama_cpp_args]
    args = [arg for arg in llama_cpp_args if not arg.startswith("--embedding")]
    return command, args
