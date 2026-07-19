"""Black Forest Labs FLUX API v1 wire models.

Docs: https://docs.bfl.ai (OpenAPI: https://api.bfl.ai/openapi.json)
Verified against the official OpenAPI spec and the legacy httpx-based engine on API_PINNED.
"""

from pydantic import JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

# NOTE: the legacy engine targeted api.us1.bfl.ai, which is dead (TLS handshake fails).
# BFL's current endpoints are the global api.bfl.ai (primary) plus regional api.us.bfl.ai
# / api.eu.bfl.ai; submission endpoints are interchangeable, but polling MUST use the
# `polling_url` returned by the submit response (BFL requirement); the engine falls back
# to BFL_GET_RESULT_URL only when the submit response omits polling_url.
BFL_API_BASE = "https://api.bfl.ai/v1"
BFL_GET_RESULT_URL = f"{BFL_API_BASE}/get_result"
BFL_POLL_INTERVAL_SECONDS = 5.0

# Terminal non-success statuses of the get_result state machine; "Pending" keeps polling.
BFL_FAILURE_STATUSES = frozenset(
    {"Task not found", "Request Moderated", "Content Moderated", "Error"}
)


class BflImageRequest(EngineRequestPayload):
    # NOTE: mirrors the legacy engine's submit payload. None fields are dropped on the
    # wire (exclude_none) — BFL 500s on explicit nulls for unsupported parameters.
    prompt: str
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    safety_tolerance: int | None = None


class BflGetResultRequest(EngineRequestPayload):
    """Empty body: the task id travels in the `id` query param, or is already embedded
    in the submit response's polling_url."""


class BflSubmitResponse(EngineResponsePayload):
    id: str
    polling_url: str | None = None


class BflPollResult(EngineResponsePayload):
    sample: str | None = None


class BflPollResponse(EngineResponsePayload):
    # NOTE: status is the get_result state machine: "Task not found" | "Pending" |
    # "Request Moderated" | "Content Moderated" | "Ready" | "Error". Kept as str (not a
    # Literal) so new provider statuses never break parsing of an otherwise valid body.
    id: str | None = None
    status: str
    result: BflPollResult | None = None
    progress: float | None = None
    details: dict[str, JsonValue] | None = None
