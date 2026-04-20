import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def _env_truthy(name: str) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _log_path() -> Path:
    explicit = (os.environ.get("SYMAI_LOG_LLM_REQUESTS_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    log_dir = (os.environ.get("SYMAI_LOG_LLM_REQUESTS_DIR") or "").strip()
    if log_dir:
        return (Path(log_dir).expanduser() / "llm_requests.jsonl")

    return Path("~/.symai/logs/llm_requests.jsonl").expanduser()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except Exception:
            try:
                return _json_safe(model_dump())
            except Exception:
                return str(value)

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            return _json_safe(as_dict())
        except Exception:
            return str(value)

    return str(value)


def _redact(obj: Any) -> Any:
    """Best-effort redaction for obvious secret keys."""

    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            lowered = key.lower()
            if lowered in {"api_key", "authorization", "x-api-key", "api-key", "bearer"}:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact(v)
        return redacted

    if isinstance(obj, list):
        return [_redact(v) for v in obj]

    return obj


def log_llm_request(*, provider: str, engine: str, model: str | None, payload: Any) -> None:
    """Opt-in logger for provider request payloads.

    Enable with:
    - SYMAI_LOG_LLM_REQUESTS=1

    Optional:
    - SYMAI_LOG_LLM_REQUESTS_PATH=/abs/or/~/path.jsonl
    - SYMAI_LOG_LLM_REQUESTS_DIR=/abs/or/~/dir  (writes llm_requests.jsonl)
    """

    if not _env_truthy("SYMAI_LOG_LLM_REQUESTS"):
        return

    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": uuid.uuid4().hex,
            "provider": provider,
            "engine": engine,
            "model": model,
            "payload": _redact(_json_safe(deepcopy(payload))),
        }

        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(record, ensure_ascii=False)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        # Never break inference because logging failed.
        return
