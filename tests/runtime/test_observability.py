import logging

import pytest

from symai.runtime.models import TokenUsage
from symai.runtime.observability import ExecutionRecord, log_executions


def execution_record(*, error: Exception | None) -> ExecutionRecord:
    return ExecutionRecord(
        engine="smart",
        capability="language_model",
        provider="openai",
        requested_model="requested-model",
        response_model="served-model",
        usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        rate_limit=None,
        request_id="request-1",
        status_code=200,
        duration_s=0.25,
        error=error,
    )


def test_log_executions_emits_structured_success_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="symai.runtime"):
        log_executions(execution_record(error=None))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "engine=smart model=served-model ok"
    assert record.engine == "smart"  # type: ignore[attr-defined]
    assert record.provider == "openai"  # type: ignore[attr-defined]
    assert record.request_id == "request-1"  # type: ignore[attr-defined]
    assert record.duration_s == 0.25  # type: ignore[attr-defined]
    assert record.prompt_tokens == 8  # type: ignore[attr-defined]
    assert record.completion_tokens == 3  # type: ignore[attr-defined]


def test_log_executions_emits_structured_failure_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("failed")

    with caplog.at_level(logging.ERROR, logger="symai.runtime"):
        log_executions(execution_record(error=error))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "engine=smart model=served-model failed"
    assert record.status == 200  # type: ignore[attr-defined]
    assert not hasattr(record, "prompt")
