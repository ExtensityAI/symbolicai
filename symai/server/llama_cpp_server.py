import argparse
import subprocess
import sys
import os
from loguru import logger

def llama_cpp_server():
    parser = argparse.ArgumentParser(description="A wrapper for llama_cpp.", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show available options for llama_cpp server.")
    parser.add_argument("--env", choices=['python', 'cpp'], default='python',
                      help="Choose programming environment (python or cpp)")
    parser.add_argument("--cpp-server-path", type=str, help="Path to llama.cpp server executable")

    main_args, llama_cpp_args = parser.parse_known_args()

    if main_args.help:
        if main_args.env == 'python':
            command = [sys.executable, "-m", "llama_cpp.server", "--help"]
            subprocess.run(command)
        else:
            if not main_args.cpp_server_path:
                logger.error("Error: --cpp-server-path is required when using cpp environment")
                sys.exit(1)
            if not os.path.exists(main_args.cpp_server_path):
                logger.error(f"Error: Executable not found at {main_args.cpp_server_path}")
                sys.exit(1)
            command = [main_args.cpp_server_path, "--help"]
            subprocess.run(command)
        sys.exit(0)

    if main_args.env == 'cpp':
        if not main_args.cpp_server_path:
            logger.error("Error: --cpp-server-path is required when using cpp environment")
            sys.exit(1)
        if not os.path.exists(main_args.cpp_server_path):
            logger.error(f"Error: Executable not found at {main_args.cpp_server_path}")
            sys.exit(1)
        command = [
            main_args.cpp_server_path,
            *llama_cpp_args,
        ]
        llama_cpp_args = [arg for arg in llama_cpp_args if not arg.startswith("--embedding")] # Exclude embedding argument
    else:  # python
        command = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            *llama_cpp_args,
        ]

    return command, llama_cpp_args
