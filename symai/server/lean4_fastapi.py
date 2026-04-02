"""Lean4 Server - FastAPI server for Lean4 code execution with Docker container management."""

from __future__ import annotations

import atexit
import os
import shlex
import time
from typing import Any

import docker
import docker.errors
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="Lean4 Server", version="1.0.0")

# Container manager singleton
_container_manager: ContainerManager | None = None

LEAN_PATH = "/usr/local/elan/bin/lean"


class LeanRequest(BaseModel):
    """Request to check/execute Lean code."""

    code: str


class LeanResponse(BaseModel):
    """Response from Lean execution."""

    status: str
    output: str
    execution_time: float


class ContainerManager:
    """Manages Docker container lifecycle with idle timeout.

    Uses `docker exec` (not SSH) for code execution, eliminating paramiko/SSH complexity.
    """

    IDLE_TIMEOUT = 300  # 5 minutes
    CONTAINER_NAME = "lean4-server-container"
    IMAGE_NAME = "lean4-container-image"

    def __init__(self):
        self.container: Any = None
        self.docker_client: Any = None
        self.last_used: float = 0
        self._exec_counter = 0
        self._init_docker()
        atexit.register(self.cleanup)

    def _init_docker(self) -> None:
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException as e:
            msg = f"Failed to initialize Docker client: {e}"
            raise RuntimeError(msg) from e

    def ensure_container(self) -> Any:
        """Ensure container exists and is running."""
        current_time = time.time()

        # Check if tracked container is still usable
        if self.container:
            if current_time - self.last_used > self.IDLE_TIMEOUT:
                logger.info("Container idle timeout reached, recreating...")
                self.cleanup()
            else:
                try:
                    self.container.reload()
                    if self.container.status == "running":
                        self.last_used = current_time
                        return self.container
                    logger.info("Container not running, recreating...")
                    self.cleanup()
                except docker.errors.APIError:
                    logger.info("Container check failed, recreating...")
                    self.cleanup()

        # Check for existing container by name (e.g. started by docker-compose)
        try:
            existing = self.docker_client.containers.get(self.CONTAINER_NAME)
            if existing.status == "running":
                logger.info(f"Reusing existing container '{self.CONTAINER_NAME}'")
                self.container = existing
                self.last_used = current_time
                return self.container
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # No existing container; will create a new one below

        # Create new container
        logger.info("Creating new Lean4 container...")
        self.container = self._create_container()
        self.last_used = current_time
        return self.container

    def _create_container(self) -> Any:
        """Create and start Docker container."""
        try:
            old = self.docker_client.containers.get(self.CONTAINER_NAME)
            old.remove(force=True)
            logger.info(f"Removed old container '{self.CONTAINER_NAME}'")
        except docker.errors.NotFound:
            pass  # No old container to remove

        container = self.docker_client.containers.run(
            self.IMAGE_NAME,
            detach=True,
            name=self.CONTAINER_NAME,
        )
        logger.info(f"Created container '{container.id[:12]}'")

        # Wait for container to be ready
        time.sleep(1)
        return container

    def execute_lean(self, code: str) -> tuple[str, str]:
        """Execute Lean code in container via a single docker exec call."""
        container = self.ensure_container()

        # PID + counter avoids collisions across workers and within the same second
        self._exec_counter += 1
        remote_path = f"/tmp/lean_{os.getpid()}_{self._exec_counter}.lean"

        # Write file and execute lean in a single exec_run call
        escaped = shlex.quote(code)
        result = container.exec_run(
            ["bash", "-c",
             f"printf '%s' {escaped} > {remote_path}"
             f" && {LEAN_PATH} {remote_path} 2>&1"
             f"; ec=$?; rm -f {remote_path}; exit $ec"],
        )

        output = result.output.decode() if isinstance(result.output, bytes) else str(result.output)
        status = "failure" if result.exit_code != 0 else "success"

        self.last_used = time.time()
        return status, output

    def cleanup(self) -> None:
        """Kill container and cleanup resources."""
        if self.container:
            try:
                logger.info(f"Killing container '{self.container.id[:12]}'")
                self.container.remove(force=True)
            except docker.errors.APIError as e:
                logger.warning(f"Error removing container: {e}")
            finally:
                self.container = None

    def health(self) -> dict:
        """Health check."""
        if not self.container:
            return {"status": "no_container", "idle_time": None}

        try:
            self.container.reload()
            if self.container.status == "running":
                idle_time = time.time() - self.last_used
                return {"status": "running", "idle_time": idle_time}
            return {"status": self.container.status, "idle_time": None}
        except docker.errors.APIError as e:
            logger.error(f"Docker API error during health check: {e}")
            return {"status": "error", "idle_time": None}


def get_container_manager() -> ContainerManager:
    """Get or create container manager singleton."""
    global _container_manager
    if _container_manager is None:
        _container_manager = ContainerManager()
    return _container_manager


# Plain `def` so FastAPI auto-runs it in a threadpool (execute_lean is blocking I/O)
@app.post("/check", response_model=LeanResponse)
def check_lean(request: LeanRequest) -> LeanResponse:
    """Check/execute Lean code."""
    manager = get_container_manager()
    start_time = time.time()

    try:
        status, output = manager.execute_lean(request.code)
        execution_time = time.time() - start_time

        return LeanResponse(
            status=status,
            output=output,
            execution_time=execution_time,
        )
    except (docker.errors.DockerException, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    manager = get_container_manager()
    return manager.health()


@app.post("/cleanup")
def cleanup() -> dict:
    """Force container cleanup."""
    manager = get_container_manager()
    manager.cleanup()
    return {"status": "cleaned"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Lean4 Server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
