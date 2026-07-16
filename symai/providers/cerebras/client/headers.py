import httpx

from symai.providers._http.headers import (
    extract_response_metadata as extract_base_response_metadata,
)
from symai.providers._http.headers import parse_optional_float, parse_optional_int
from symai.providers.cerebras.client.response import HttpMetadata, RateLimitState

REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"
LIMIT_REQUESTS_DAY_HEADER = "x-ratelimit-limit-requests-day"
LIMIT_TOKENS_MINUTE_HEADER = "x-ratelimit-limit-tokens-minute"
REMAINING_REQUESTS_DAY_HEADER = "x-ratelimit-remaining-requests-day"
REMAINING_TOKENS_MINUTE_HEADER = "x-ratelimit-remaining-tokens-minute"
RESET_REQUESTS_DAY_HEADER = "x-ratelimit-reset-requests-day"
RESET_TOKENS_MINUTE_HEADER = "x-ratelimit-reset-tokens-minute"


def extract_response_metadata(response: httpx.Response) -> HttpMetadata:
    base_metadata = extract_base_response_metadata(response)
    headers = response.headers
    return HttpMetadata(
        status_code=base_metadata.status_code,
        request_id=base_metadata.request_id,
        retry_after=base_metadata.retry_after,
        rate_limit=RateLimitState(
            limit_requests_day=parse_optional_int(headers.get(LIMIT_REQUESTS_DAY_HEADER)),
            limit_tokens_minute=parse_optional_int(headers.get(LIMIT_TOKENS_MINUTE_HEADER)),
            remaining_requests_day=parse_optional_int(headers.get(REMAINING_REQUESTS_DAY_HEADER)),
            remaining_tokens_minute=parse_optional_int(headers.get(REMAINING_TOKENS_MINUTE_HEADER)),
            reset_requests_day=parse_optional_float(headers.get(RESET_REQUESTS_DAY_HEADER)),
            reset_tokens_minute=parse_optional_float(headers.get(RESET_TOKENS_MINUTE_HEADER)),
        ),
    )
