import symai.providers.cerebras.client.chat as chat
import symai.providers.cerebras.client.errors as errors
from symai.providers._http.client import BaseClient, ClientConfig
from symai.providers.cerebras.client.headers import extract_response_metadata
from symai.providers.cerebras.client.response import APIResponse, HttpMetadata

BASE_URL = "https://api.cerebras.ai/v1"

CLIENT_CONFIG = ClientConfig[HttpMetadata](
    base_url=BASE_URL,
    provider_name="Cerebras",
    extract_response_metadata=extract_response_metadata,
    api_error=errors.APIError,
    auth_error=errors.AuthError,
    rate_limit_error=errors.RateLimitError,
    response_error=errors.ResponseError,
    transport_error=errors.TransportError,
)


class Client(BaseClient[HttpMetadata]):
    """Synchronous owner of a Cerebras HTTP connection pool."""

    config = CLIENT_CONFIG

    def create_chat_completion(
        self,
        request: chat.CreateChatCompletionRequest,
    ) -> APIResponse[chat.ChatCompletion, HttpMetadata]:
        """Execute one non-streaming chat completion request."""

        return self._post(chat.PATH, request, chat.ChatCompletion)
