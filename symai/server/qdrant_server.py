import argparse
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

from loguru import logger


class QdrantConfigFlag(str, Enum):
    """String/int flags persisted to symserver.config.json.

    Enum name == argparse dest attr; enum value == CLI flag string.
    Iterated with `is not None` to preserve int 0 (e.g. --max-workers 0).
    """

    api_key            = "--api-key"
    read_only_api_key  = "--read-only-api-key"
    log_level          = "--log-level"
    max_workers        = "--max-workers"
    max_search_threads = "--max-search-threads"
    snapshots_path     = "--snapshots-path"


class QdrantTLSFlag(str, Enum):
    """TLS certificate path flags. name == argparse dest; value == CLI flag."""

    tls_cert    = "--tls-cert"
    tls_key     = "--tls-key"
    tls_ca_cert = "--tls-ca-cert"


class QdrantDockerFlag(str, Enum):
    """Docker container management flags. name == argparse dest; value == CLI flag."""

    docker_image     = "--docker-image"
    docker_container = "--docker-container-name"
    docker_detach    = "--docker-detach"
    docker_remove    = "--docker-remove"


class QdrantStorageFlag(str, Enum):
    """Server storage and execution path flags. name == argparse dest; value == CLI flag."""

    storage_path    = "--storage-path"
    use_env_storage = "--use-env-storage"
    binary_path     = "--binary-path"


class QdrantBoolFlag(str, Enum):
    """Boolean (store_true) flags. name == argparse dest; value == CLI flag."""

    disable_telemetry = "--disable-telemetry"
    enable_tls        = "--enable-tls"


class QdrantGenericFlag(str, Enum):
    """Generic QDRANT__* env-var passthrough.

    Note: `--set` uses dest='qdrant_set', not 'set', so the name==dest
    convention does not hold here. This flag is only used for routing detection.
    """

    set = "--set"


# QdrantServerFlag is the union of all qdrant_server.py-exclusive CLI flags,
# composed from the sub-enums above with no additional inline entries.
_FLAG_CATEGORIES = (
    QdrantConfigFlag,
    QdrantTLSFlag,
    QdrantDockerFlag,
    QdrantStorageFlag,
    QdrantBoolFlag,
    QdrantGenericFlag,
)
QdrantServerFlag = Enum(
    "QdrantServerFlag",
    {m.name: m.value for cat in _FLAG_CATEGORIES for m in cat},
    type=str,
)
del _FLAG_CATEGORIES
_QDRANT_SERVER_FLAGS = frozenset(QdrantServerFlag)


def _build_qdrant_env(args) -> dict:
    """Collect QDRANT__* env vars from parsed CLI args.

    TLS cert paths are intentionally excluded: binary mode sets host paths
    explicitly after this call; Docker mode sets container paths after mounting
    the cert directory as a volume.
    """
    env = {}
    if args.api_key:
        env["QDRANT__SERVICE__API_KEY"] = args.api_key
    if args.read_only_api_key:
        env["QDRANT__SERVICE__READ_ONLY_API_KEY"] = args.read_only_api_key
    if args.log_level:
        env["QDRANT__LOG_LEVEL"] = args.log_level
    if args.max_workers is not None:
        env["QDRANT__SERVICE__MAX_WORKERS"] = str(args.max_workers)
    if args.max_search_threads is not None:
        env["QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS"] = str(args.max_search_threads)
    if args.disable_telemetry:
        env["QDRANT__TELEMETRY_DISABLED"] = "true"
    for kv in args.qdrant_set:
        key, _, value = kv.partition("=")
        key = key.strip()
        if not key.startswith("QDRANT__"):
            key = f"QDRANT__{key}"
        env[key] = value.strip()
    return env


def _apply_tls_env(env, args, path_transform=lambda p: p):
    """Populate TLS env vars into env, applying path_transform to cert file paths.

    Binary mode passes the identity transform (host paths used as-is).
    Docker mode passes a transform that rewrites to the container mount path.
    """
    if args.enable_tls:
        env["QDRANT__SERVICE__ENABLE_TLS"] = "true"
    if args.tls_cert:
        env["QDRANT__TLS__CERT"] = path_transform(args.tls_cert)
    if args.tls_key:
        env["QDRANT__TLS__KEY"] = path_transform(args.tls_key)
    if args.tls_ca_cert:
        env["QDRANT__TLS__CA_CERT"] = path_transform(args.tls_ca_cert)


def qdrant_server():
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
    # --- Security / Auth ---
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Qdrant API key required on every request (QDRANT__SERVICE__API_KEY)",
    )
    parser.add_argument(
        "--read-only-api-key",
        type=str,
        default=None,
        help="Read-only API key (QDRANT__SERVICE__READ_ONLY_API_KEY)",
    )
    # --- Performance ---
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel HTTP request workers in Qdrant (QDRANT__SERVICE__MAX_WORKERS; 0=auto)",
    )
    parser.add_argument(
        "--max-search-threads",
        type=int,
        default=None,
        help="Search threads per request (QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS; 0=auto)",
    )
    # --- Observability ---
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARN", "ERROR"],
        default=None,
        help="Log verbosity level (QDRANT__LOG_LEVEL)",
    )
    parser.add_argument(
        "--disable-telemetry",
        action="store_true",
        default=False,
        help="Opt out of Qdrant usage telemetry (QDRANT__TELEMETRY_DISABLED)",
    )
    # --- Storage ---
    parser.add_argument(
        "--snapshots-path",
        type=str,
        default=None,
        help="Directory for Qdrant snapshots (QDRANT__STORAGE__SNAPSHOTS_PATH)",
    )
    # --- TLS ---
    parser.add_argument(
        "--enable-tls",
        action="store_true",
        default=False,
        help="Enable HTTPS on REST and gRPC (QDRANT__SERVICE__ENABLE_TLS)",
    )
    parser.add_argument(
        "--tls-cert",
        type=str,
        default=None,
        help="Path to TLS certificate file; all cert files must share the same directory",
    )
    parser.add_argument("--tls-key", type=str, default=None, help="Path to TLS private key file")
    parser.add_argument(
        "--tls-ca-cert", type=str, default=None, help="Path to TLS CA certificate file"
    )
    # --- Generic QDRANT__* passthrough ---
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="qdrant_set",
        help="Set any QDRANT__* env var directly, e.g. --set SERVICE__CORS=false. "
        "The QDRANT__ prefix is added automatically if omitted.",
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
            os.environ["QDRANT__SERVICE__ENABLE_STATIC_CONTENT_CACHE"] = "false"

        # Apply QDRANT__* env vars via os.environ (binary reads from environment)
        qdrant_env = _build_qdrant_env(main_args)
        if main_args.snapshots_path:
            snap = Path(main_args.snapshots_path)
            snap.mkdir(parents=True, exist_ok=True)
            qdrant_env["QDRANT__STORAGE__SNAPSHOTS_PATH"] = str(snap.resolve())
        # TLS: enable flag + cert paths together (binary reads host filesystem directly)
        _apply_tls_env(qdrant_env, main_args)
        for k, v in qdrant_env.items():
            os.environ[k] = v

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
            command.extend(["-e", "QDRANT__SERVICE__ENABLE_STATIC_CONTENT_CACHE=false"])

        # Snapshots: mount host directory, rewrite env var to container path
        if main_args.snapshots_path:
            snap = Path(main_args.snapshots_path)
            snap.mkdir(parents=True, exist_ok=True)
            command.extend(["-v", f"{snap.resolve()}:/qdrant/snapshots:z"])

        # TLS: mount the cert directory as read-only, rewrite cert paths to container paths
        if main_args.enable_tls and any(
            (main_args.tls_cert, main_args.tls_key, main_args.tls_ca_cert)
        ):
            tls_host_file = main_args.tls_cert or main_args.tls_key or main_args.tls_ca_cert
            tls_dir = Path(tls_host_file).resolve().parent
            command.extend(["-v", f"{tls_dir}:/qdrant/tls:ro"])

        # Collect QDRANT__* env vars; set path-dependent vars with container paths for Docker
        qdrant_env = _build_qdrant_env(main_args)
        if main_args.snapshots_path:
            qdrant_env["QDRANT__STORAGE__SNAPSHOTS_PATH"] = "/qdrant/snapshots"
        # TLS: enable flag + container-rewritten cert paths together
        _apply_tls_env(qdrant_env, main_args, lambda p: f"/qdrant/tls/{Path(p).name}")
        for k, v in qdrant_env.items():
            command.extend(["-e", f"{k}={v}"])

        # Docker image
        command.append(main_args.docker_image)

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

    # Persist new flags to symserver.config.json (common to both modes)
    for flag in QdrantConfigFlag:
        val = getattr(main_args, flag.name, None)
        if val is not None:
            config_args.extend([flag.value, str(val)])
    if main_args.disable_telemetry:
        config_args.append("--disable-telemetry")
    if main_args.enable_tls:
        config_args.append("--enable-tls")
    for flag in QdrantTLSFlag:
        val = getattr(main_args, flag.name, None)
        if val:
            config_args.extend([flag.value, val])
    for kv in main_args.qdrant_set:
        config_args.extend(["--set", kv])

    return command, config_args
