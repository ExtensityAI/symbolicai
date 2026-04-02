"""Lean4 Server wrapper for symserver integration.

This module provides the symserver integration for Lean4. It:
1. Checks FORMAL_ENGINE is set to "local"
2. Builds Docker image if missing
3. Starts FastAPI server with auto-increment port
"""

import argparse
import socket
import subprocess
import sys
from pathlib import Path

from loguru import logger

from symai.backend.settings import SYMAI_CONFIG

IMAGE_NAME = "lean4-container-image"


def _find_free_port() -> int:
    """Let the OS assign a free port atomically on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lean4_server() -> tuple[list[str], list[str]]:
    """Lean4 server wrapper for Docker image build + FastAPI startup."""
    parser = argparse.ArgumentParser(description="Lean4 server", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show help")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server HTTP port (falls back to OS-assigned if taken)",
    )

    args, remaining = parser.parse_known_args()

    # Filter out --lean4 from remaining args (it's a symserver flag, not for uvicorn)
    remaining = [arg for arg in remaining if arg != "--lean4"]

    if args.help:
        parser.print_help()
        logger.info("\nFastAPI server options:")
        logger.info("  --host HOST         Bind socket to this host (default: 0.0.0.0)")
        logger.info("  --reload            Enable auto-reload")
        sys.exit(0)

    if SYMAI_CONFIG.get("FORMAL_ENGINE") != "local":
        logger.error("FORMAL_ENGINE must be set to 'local' to use Lean4 server")
        logger.error("Set it in your config or run: export FORMAL_ENGINE=local")
        sys.exit(1)

    # Build image if it doesn't exist
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE_NAME],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            logger.info("Building Lean4 container image...")
            formal_dir = Path(__file__).parent.parent / "backend" / "engines" / "formal"
            subprocess.run(
                ["docker", "build", "-t", IMAGE_NAME, str(formal_dir)],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build container image: {e}")
        sys.exit(1)

    # Try requested port, fall back to OS-assigned if taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", args.port)) != 0:
            port = args.port
        else:
            port = _find_free_port()
            logger.info(f"Port {args.port} taken, using port {port}")

    # Start FastAPI server (ContainerManager inside lean4_fastapi handles Docker lifecycle)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "symai.server.lean4_fastapi:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        *remaining,
    ]

    return command, remaining
