"""Lean4 Local Engine - HTTP client for Lean4 Server.

This engine connects to a separate Lean4 Server process that manages
the Docker container lifecycle. The server can be started via:

    symserver --lean4

Or it auto-starts when the engine is first used.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from ....core import Argument

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG


class LeanResult(Result):
    """Represents the result of executing a Lean code snippet."""

    def __init__(self, value: dict[str, str]) -> None:
        super().__init__(value)
        self._value = value


class Lean4LocalEngine(Engine):
    """Engine for executing Lean code via HTTP to Lean4 Server.

    The server manages Docker container lifecycle with idle timeout.
    This engine is a simple HTTP client.

    Server discovery order:
        1. Explicit ``server_url`` constructor argument
        2. ``url`` field in symserver.config.json (written by ``symserver --lean4``)
        3. Auto-start a new server on a free port
    """

    DEFAULT_SERVER_URL = "http://localhost:8000"
    SERVER_START_TIMEOUT = 30  # seconds

    def __init__(self, server_url: str | None = None) -> None:
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.name = self.__class__.__name__

        # Bail out early if not configured as the active formal engine
        if self.config.get("FORMAL_ENGINE") != "local":
            return

        self.server_url = server_url or self._discover_server_url()
        self.server_process: subprocess.Popen | None = None

        # Ensure server is running
        self._ensure_server()

    def id(self) -> str:
        if self.config.get("FORMAL_ENGINE") == "local":
            return "formal"
        return super().id()

    def _discover_server_url(self) -> str:
        """Resolve server URL from symserver state or default."""
        if SYMSERVER_CONFIG.get("online") and SYMSERVER_CONFIG.get("url"):
            return SYMSERVER_CONFIG["url"]
        return self.DEFAULT_SERVER_URL

    def _ensure_server(self) -> None:
        """Ensure Lean4 Server is running, start if needed."""
        if self._is_server_healthy():
            return

        UserMessage("Lean4 Server not running. Starting server...")
        self._start_server()

        start_time = time.time()
        while time.time() - start_time < self.SERVER_START_TIMEOUT:
            if self._is_server_healthy():
                UserMessage("Lean4 Server is ready!")
                return
            time.sleep(0.5)

        msg = f"Lean4 Server failed to start within {self.SERVER_START_TIMEOUT}s"
        UserMessage(msg, raise_with=RuntimeError)

    def _is_server_healthy(self) -> bool:
        """Check if Lean4 Server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _start_server(self) -> None:
        """Start the Lean4 FastAPI server as a subprocess on a free port."""
        try:
            # Let the OS assign a free port to avoid conflicts
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]

            self.server_url = f"http://localhost:{port}"
            self.server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "symai.server.lean4_fastapi:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.SubprocessError) as e:
            msg = f"Failed to start Lean4 Server: {e}"
            UserMessage(msg, raise_with=RuntimeError)

    def _post_check(self, code: str) -> requests.Response:
        """POST to /check, retrying once if the server connection drops."""
        try:
            response = requests.post(
                f"{self.server_url}/check",
                json={"code": code},
                timeout=60,
            )
            response.raise_for_status()
            return response
        except requests.ConnectionError:
            self._ensure_server()
            response = requests.post(
                f"{self.server_url}/check",
                json={"code": code},
                timeout=60,
            )
            response.raise_for_status()
            return response

    def forward(self, argument: Argument) -> tuple[list[LeanResult], dict]:
        """Execute Lean code via HTTP to Lean4 Server."""
        code = argument.prop.prepared_input

        try:
            data = self._post_check(code).json()

            result = LeanResult(
                {
                    "output": data["output"],
                    "status": data["status"],
                }
            )

            metadata = {
                "status": data["status"],
                "execution_time": data.get("execution_time", 0),
            }

            return [result], metadata

        except requests.RequestException as e:
            msg = f"Lean4 Server request failed: {e}"
            UserMessage(msg)
            result = LeanResult({"output": str(e), "status": "error"})
            return [result], {"status": "error", "message": str(e)}

    def prepare(self, argument: Argument) -> None:
        """Prepare the input for Lean execution."""
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def cleanup(self) -> None:
        """Cleanup server process if we started it."""
        if self.server_process:
            with contextlib.suppress(requests.RequestException):
                requests.post(f"{self.server_url}/cleanup", timeout=5)
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.server_process = None
