import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger


def qdrant_server():  # noqa
    """
    A wrapper for Qdrant server that supports both Docker and binary execution modes.

    Returns:
        tuple: (command, args) where command is the list to execute and args are the parsed arguments
    """
    parser = argparse.ArgumentParser(description="A wrapper for Qdrant server.", add_help=False)
    parser.add_argument(
        "--help", action="store_true", help="Show available options for Qdrant server."
    )
    parser.add_argument(
        "--env",
        choices=["docker", "binary"],
        default="docker",
        help="Choose execution environment (docker or binary)",
    )
    parser.add_argument("--binary-path", type=str, help="Path to Qdrant binary executable")
    parser.add_argument(
        "--docker-image",
        type=str,
        default="qdrant/qdrant:latest",
        help="Docker image to use (default: qdrant/qdrant:latest)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=6333, help="REST API port (default: 6333)")
    parser.add_argument("--grpc-port", type=int, default=6334, help="gRPC API port (default: 6334)")
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./qdrant_storage",
        help="Path to Qdrant storage directory (default: ./qdrant_storage)",
    )
    parser.add_argument(
        "--use-env-storage",
        action="store_true",
        default=False,
        help="Use QDRANT__STORAGE__STORAGE_PATH environment variable instead of passing --storage-path. "
        "If set, storage path argument/volume mount will be skipped, allowing Qdrant to use its own defaults or env vars.",
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to Qdrant configuration file"
    )
    parser.add_argument(
        "--docker-container-name",
        type=str,
        default="qdrant",
        help="Name for Docker container (default: qdrant)",
    )
    parser.add_argument(
        "--docker-remove",
        action="store_true",
        default=True,
        help="Remove container when it stops (default: True)",
    )
    parser.add_argument(
        "--docker-detach",
        action="store_true",
        default=False,
        help="Run Docker container in detached mode (default: False)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable caching in Qdrant server (default: False)",
    )

    main_args, qdrant_args = parser.parse_known_args()

    if main_args.help:
        if main_args.env == "docker":
            # Show Docker help
            command = ["docker", "run", "--rm", main_args.docker_image, "--help"]
            subprocess.run(command, check=False)
        else:
            if not main_args.binary_path:
                logger.error("Error: --binary-path is required when using binary environment")
                sys.exit(1)
            if not Path(main_args.binary_path).exists():
                logger.error(f"Error: Binary not found at {main_args.binary_path}")
                sys.exit(1)
            command = [main_args.binary_path, "--help"]
            subprocess.run(command, check=False)
        sys.exit(0)

    if main_args.env == "binary":
        if not main_args.binary_path:
            logger.error("Error: --binary-path is required when using binary environment")
            sys.exit(1)
        if not Path(main_args.binary_path).exists():
            logger.error(f"Error: Binary not found at {main_args.binary_path}")
            sys.exit(1)

        # Build command for binary execution
        command = [main_args.binary_path]

        # Add storage path argument unless --use-env-storage is set
        if not main_args.use_env_storage:
            # Ensure storage directory exists
            storage_path = Path(main_args.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            abs_storage_path = str(storage_path.resolve())
            # Qdrant binary accepts --storage-path argument
            command.extend(["--storage-path", abs_storage_path])
        elif os.getenv("QDRANT__STORAGE__STORAGE_PATH"):
            # If using env storage and env var is set, pass it through
            # Note: Qdrant binary may read this from env, but we can also pass it explicitly
            abs_storage_path = os.getenv("QDRANT__STORAGE__STORAGE_PATH")
            command.extend(["--storage-path", abs_storage_path])

        # Add host, port, and grpc-port arguments
        command.extend(["--host", main_args.host])
        command.extend(["--port", str(main_args.port)])
        command.extend(["--grpc-port", str(main_args.grpc_port)])

        if main_args.config_path:
            command.extend(["--config-path", main_args.config_path])

        # Add no-cache environment variable if flag is set
        if main_args.no_cache:
            # Set environment variable to disable caching
            # Qdrant uses environment variables with QDRANT__ prefix
            os.environ["QDRANT__SERVICE__ENABLE_STATIC_CONTENT_CACHE"] = "false"

        # Add any additional Qdrant-specific arguments
        command.extend(qdrant_args)

    else:  # docker
        # Build Docker command
        command = ["docker", "run"]

        # Container management options
        if main_args.docker_remove:
            command.append("--rm")

        if main_args.docker_detach:
            command.append("-d")
        # Note: We don't add -it by default to avoid issues in non-interactive environments
        # Users can add it manually if needed via qdrant_args

        # Container name
        command.extend(["--name", main_args.docker_container_name])

        # Port mappings
        command.extend(["-p", f"{main_args.port}:6333"])
        command.extend(["-p", f"{main_args.grpc_port}:6334"])

        # Volume mount for storage (skip if --use-env-storage is set)
        if not main_args.use_env_storage:
            # Ensure storage directory exists
            storage_path = Path(main_args.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            abs_storage_path = str(storage_path.resolve())
            # Volume mount for storage
            command.extend(["-v", f"{abs_storage_path}:/qdrant/storage:z"])
            # Set storage path environment variable to use the mounted volume
            command.extend(["-e", "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage"])
        elif os.getenv("QDRANT__STORAGE__STORAGE_PATH"):
            # If using env storage and env var is set, pass it through to container
            env_storage_path = os.getenv("QDRANT__STORAGE__STORAGE_PATH")
            command.extend(["-e", f"QDRANT__STORAGE__STORAGE_PATH={env_storage_path}"])

        # Volume mount for config (if provided)
        # Note: Qdrant Docker image accepts environment variables and config files
        # For custom config, mount it as a volume before the image name
        if main_args.config_path:
            config_path = Path(main_args.config_path)
            abs_config_path = config_path.resolve()
            config_dir = str(abs_config_path.parent)
            command.extend(["-v", f"{config_dir}:/qdrant/config:z"])
            # Qdrant looks for config.yaml in /qdrant/config by default

        # Add no-cache environment variable if flag is set
        if main_args.no_cache:
            # Set environment variable to disable caching in Docker container
            command.extend(["-e", "QDRANT__SERVICE__ENABLE_STATIC_CONTENT_CACHE=false"])

        # Docker image
        command.append(main_args.docker_image)

        # Qdrant server arguments (if any additional ones are passed)

        # Add any additional Qdrant arguments
        if qdrant_args:
            command.extend(qdrant_args)

    # Prepare args for config storage (similar to llama_cpp_server pattern)
    # Extract key-value pairs for configuration
    config_args = []
    if main_args.env == "docker":
        config_args = [
            "--env",
            main_args.env,
            "--host",
            main_args.host,
            "--port",
            str(main_args.port),
            "--grpc-port",
            str(main_args.grpc_port),
            "--docker-image",
            main_args.docker_image,
            "--docker-container-name",
            main_args.docker_container_name,
        ]
        # Only include storage-path in config if not using env storage
        if not main_args.use_env_storage:
            config_args.extend(["--storage-path", main_args.storage_path])
        else:
            config_args.append("--use-env-storage")
        if main_args.config_path:
            config_args.extend(["--config-path", main_args.config_path])
        if main_args.no_cache:
            config_args.append("--no-cache")
    else:
        config_args = [
            "--env",
            main_args.env,
            "--binary-path",
            main_args.binary_path,
            "--host",
            main_args.host,
            "--port",
            str(main_args.port),
            "--grpc-port",
            str(main_args.grpc_port),
        ]
        # Only include storage-path in config if not using env storage
        if not main_args.use_env_storage:
            config_args.extend(["--storage-path", main_args.storage_path])
        else:
            config_args.append("--use-env-storage")
        if main_args.config_path:
            config_args.extend(["--config-path", main_args.config_path])
        if main_args.no_cache:
            config_args.append("--no-cache")

    return command, config_args
