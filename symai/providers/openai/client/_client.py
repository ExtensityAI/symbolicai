import symai.providers.openai.client.embeddings as embeddings
import symai.providers.openai.client.errors as errors
import symai.providers.openai.client.responses as responses_api
from symai.providers._client.client import BaseClient, ClientConfig
from symai.providers.openai.client.headers import extract_response_metadata
from symai.providers.openai.client.transport import APIResponse, ResponseMetadata

BASE_URL = "https://api.openai.com/v1"

CLIENT_CONFIG = ClientConfig[ResponseMetadata](
    base_url=BASE_URL,
    provider_name="OpenAI",
    extract_response_metadata=extract_response_metadata,
    api_error=errors.APIError,
    auth_error=errors.AuthError,
    rate_limit_error=errors.RateLimitError,
    response_error=errors.ResponseError,
    transport_error=errors.TransportError,
)


class Client(BaseClient[ResponseMetadata]):
    """Synchronous owner of an OpenAI HTTP connection pool."""

    config = CLIENT_CONFIG

    def create_response(
        self,
        request: responses_api.CreateResponseRequest,
    ) -> APIResponse[responses_api.Response, ResponseMetadata]:
        return self._post(
            responses_api.PATH,
            request,
            responses_api.Response,
        )

    def create_embeddings(
        self,
        request: embeddings.CreateEmbeddingRequest,
    ) -> APIResponse[embeddings.EmbeddingList, ResponseMetadata]:
        return self._post(embeddings.PATH, request, embeddings.EmbeddingList)
