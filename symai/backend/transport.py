from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from symai.backend.request import EngineAPIRequest

DEFAULT_RETRIES = 2
RETRY_STATUS_CODES = frozenset({408, 409, 429, 529})
ENGINE_API_CLIENT = httpx.Client(
    timeout=None,
    # NOTE: These connection-pool limits mirror OpenAI's Python SDK defaults.
    # TODO: Expose custom httpx.Client injection for workloads that need a different pool.
    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
)


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str
    id: str | None = None
    retry_ms: int | None = None


class EngineAPIError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str | None,
        message: str,
        request_id: str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


class EngineAuthenticationError(EngineAPIError):
    """The provider rejected credentials (401) or denied access (403)."""


class EngineRateLimitError(EngineAPIError):
    """The provider throttled the request (429)."""

    def __init__(self, *, retry_after: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.retry_after = retry_after


class EngineTransportError(httpx.RequestError):
    """The request never produced a well-formed response (connection, DNS, TLS, timeout)
    and retries are exhausted. Subclasses httpx.RequestError so existing handlers keep
    catching it."""


def default_engine_api_client() -> httpx.Client:
    return ENGINE_API_CLIENT


def build_request_options(request: EngineAPIRequest) -> dict[str, Any]:
    options = {
        "headers": request.headers,
        "params": request.params,
    }
    if request.files is not None:
        # NOTE: multipart form — scalar payload fields ride as form data, binaries in
        # `files`; httpx sets the multipart Content-Type (with boundary) itself.
        options["data"] = {
            key: value if isinstance(value, str) else str(value)
            for key, value in request.body().items()
        }
        options["files"] = request.files
    elif request.method.upper() != "GET":
        # NOTE: GET APIs (Wolfram, BFL poll) carry their payload in `params`; a JSON
        # body on GET is at best ignored and at worst rejected.
        options["json"] = request.body()
    if request.timeout is not None:
        options["timeout"] = request.timeout
    return options


def execute_engine_api_request(
    request: EngineAPIRequest,
    *,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_RETRIES,
) -> httpx.Response:
    active_client = default_engine_api_client() if client is None else client
    request_options = build_request_options(request)
    attempt = 0

    while True:
        try:
            response = active_client.request(request.method, request.url, **request_options)
        except httpx.RequestError as exc:
            if attempt >= max_retries:
                raise EngineTransportError(str(exc), request=exc.request) from exc
            time.sleep(retry_delay_seconds(attempt, None))
            attempt += 1
            continue

        if not should_retry_response(response) or attempt >= max_retries:
            if response.is_error:
                raise build_engine_api_error(response)
            return response

        time.sleep(retry_delay_seconds(attempt, response))
        attempt += 1


def execute_engine_api_stream_events(
    request: EngineAPIRequest,
    *,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_RETRIES,
) -> Iterator[SSEEvent]:
    active_client = default_engine_api_client() if client is None else client
    request_options = build_request_options(request)
    attempt = 0

    # NOTE: Retries stop once response body iteration starts because the provider may
    # have generated output before a complete SSE event is yielded.
    while True:
        retry_response = None
        streaming_started = False
        try:
            with active_client.stream(request.method, request.url, **request_options) as response:
                if should_retry_response(response) and attempt < max_retries:
                    response.read()
                    retry_response = response
                elif response.is_error:
                    response.read()
                    raise build_engine_api_error(response)
                else:
                    streaming_started = True
                    yield from iter_sse_events(response)
                    return
        except httpx.RequestError as exc:
            if streaming_started or attempt >= max_retries:
                raise EngineTransportError(str(exc), request=exc.request) from exc

        time.sleep(retry_delay_seconds(attempt, retry_response))
        attempt += 1


def iter_sse_events(response: httpx.Response) -> Iterator[SSEEvent]:
    event = None
    event_id = None
    retry_ms = None
    data_lines = []
    has_fields = False

    for line in response.iter_lines():
        if line == "":
            if has_fields:
                yield SSEEvent(
                    event=event,
                    data="\n".join(data_lines),
                    id=event_id,
                    retry_ms=retry_ms,
                )
            event = None
            event_id = None
            retry_ms = None
            data_lines = []
            has_fields = False
            continue

        if line.startswith(":"):
            continue

        field_name, field_value = parse_sse_line(line)
        has_fields = True
        if field_name == "data":
            data_lines.append(field_value)
        elif field_name == "event":
            event = field_value
        elif field_name == "id":
            event_id = field_value
        elif field_name == "retry":
            retry_ms = parse_sse_retry_ms(field_value)

    if has_fields:
        yield SSEEvent(
            event=event,
            data="\n".join(data_lines),
            id=event_id,
            retry_ms=retry_ms,
        )


def parse_sse_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return line, ""
    field_name, field_value = line.split(":", 1)
    if field_value.startswith(" "):
        field_value = field_value[1:]
    return field_name, field_value


def parse_sse_retry_ms(value: str) -> int | None:
    try:
        retry_ms = int(value)
    except ValueError:
        return None
    if retry_ms < 0:
        return None
    return retry_ms


def should_retry_response(response: httpx.Response) -> bool:
    if response.headers.get("x-should-retry") == "false":
        return False
    if response.headers.get("x-should-retry") == "true":
        return True
    return response.status_code in RETRY_STATUS_CODES or response.status_code >= 500


def retry_delay_seconds(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = None
            if delay is not None and 0 <= delay <= 60:
                return delay
    return min(0.5 * (2**attempt), 8.0)


def build_engine_api_error(response: httpx.Response) -> EngineAPIError:
    try:
        data = response.json()
    except ValueError:
        data = {}

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message") or response.text
    else:
        code = None
        message = response.text

    kwargs = {
        "status_code": response.status_code,
        "code": str(code) if code is not None else None,
        "message": str(message),
        "request_id": response.headers.get("x-request-id"),
    }
    if response.status_code in (401, 403):
        return EngineAuthenticationError(**kwargs)
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        try:
            retry_after_seconds = float(retry_after) if retry_after is not None else None
        except ValueError:
            retry_after_seconds = None
        return EngineRateLimitError(retry_after=retry_after_seconds, **kwargs)
    return EngineAPIError(**kwargs)
