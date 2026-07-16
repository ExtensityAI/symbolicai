import logging
from collections.abc import Callable
from dataclasses import dataclass

from symai.runtime.errors import EngineCapability, ExecutionError, SymbolicAIRuntimeError
from symai.runtime.models import (
    EmbeddingResponse,
    LanguageModelResponse,
    RateLimitMetadata,
    TokenUsage,
)

logger = logging.getLogger("symai.runtime")


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    engine: str
    capability: EngineCapability
    provider: str | None
    requested_model: str | None
    response_model: str | None
    usage: TokenUsage | None
    rate_limit: RateLimitMetadata | None
    request_id: str | None
    status_code: int | None
    duration_s: float
    error: Exception | None


type Observer = Callable[[ExecutionRecord], None]


def log_executions(record: ExecutionRecord) -> None:
    model = record.response_model or record.requested_model
    extra = {
        "engine": record.engine,
        "provider": record.provider,
        "request_id": record.request_id,
        "status": record.status_code,
        "duration_s": record.duration_s,
    }
    if record.error is not None:
        logger.error(
            "engine=%s model=%s failed",
            record.engine,
            model,
            extra=extra,
        )
        return

    usage = record.usage
    extra.update(
        {
            "prompt_tokens": usage.prompt_tokens if usage is not None else None,
            "completion_tokens": usage.completion_tokens if usage is not None else None,
        }
    )
    logger.info(
        "engine=%s model=%s ok",
        record.engine,
        model,
        extra=extra,
    )


def _record_from_response(
    engine: str,
    capability: EngineCapability,
    response: LanguageModelResponse | EmbeddingResponse,
    duration_s: float,
) -> ExecutionRecord:
    metadata = response.metadata
    return ExecutionRecord(
        engine=engine,
        capability=capability,
        provider=metadata.provider,
        requested_model=metadata.requested_model,
        response_model=metadata.response_model,
        usage=metadata.usage,
        rate_limit=metadata.rate_limit,
        request_id=metadata.request_id,
        status_code=metadata.status_code,
        duration_s=duration_s,
        error=None,
    )


def _record_from_error(
    engine: str,
    capability: EngineCapability,
    error: SymbolicAIRuntimeError,
    duration_s: float,
) -> ExecutionRecord:
    metadata = error.metadata if isinstance(error, ExecutionError) else None
    return ExecutionRecord(
        engine=engine,
        capability=capability,
        provider=metadata.provider if metadata is not None else None,
        requested_model=metadata.model if metadata is not None else None,
        response_model=None,
        usage=None,
        rate_limit=None,
        request_id=metadata.request_id if metadata is not None else None,
        status_code=None,
        duration_s=duration_s,
        error=error,
    )
