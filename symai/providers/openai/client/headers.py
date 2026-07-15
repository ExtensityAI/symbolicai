import httpx

from symai.providers._client.headers import parse_optional_float
from symai.providers.openai.client.transport import ResponseMetadata

REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"


def extract_response_metadata(response: httpx.Response) -> ResponseMetadata:
    return ResponseMetadata(
        status_code=response.status_code,
        request_id=response.headers.get(REQUEST_ID_HEADER),
        retry_after=parse_optional_float(response.headers.get(RETRY_AFTER_HEADER)),
    )
