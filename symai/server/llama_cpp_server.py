import argparse
import subprocess
import sys


def llama_cpp_server():
    parser = argparse.ArgumentParser(description="A wrapper for llama_cpp.", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show available options for llama_cpp server.")

    main_args, llama_cpp_args, = parser.parse_known_args()

    if main_args.help:
        command = [sys.executable, "-m", "llama_cpp.server", "--help"]
        subprocess.run(command)
        sys.exit(0)

    command = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        *llama_cpp_args,
    ]

    return command, llama_cpp_args

