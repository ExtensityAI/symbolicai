from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from symai.providers._client.client import BaseClient
from symai.providers.cerebras.client.client import Client as CerebrasClient
from symai.providers.deepseek.client.client import Client as DeepSeekClient
from symai.providers.openai.client.client import Client as OpenAIClient


class CountingTransport(httpx.BaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.close_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.parametrize("client_type", [OpenAIClient, DeepSeekClient, CerebrasClient])
def test_provider_clients_are_endpoint_subclasses_of_shared_base(client_type):
    assert issubclass(client_type, BaseClient)
    assert client_type.__module__.startswith("symai.providers.")
    assert client_type.__module__.endswith(".client.client")


@pytest.mark.parametrize("client_type", [OpenAIClient, DeepSeekClient, CerebrasClient])
def test_injected_transport_rejects_connect_retries_without_taking_ownership(
    client_type,
):
    transport = CountingTransport(lambda _request: httpx.Response(200))

    with pytest.raises(
        ValueError,
        match=r"^connect_retries cannot be combined with an injected transport$",
    ):
        client_type(
            api_key=SecretStr("test-key"),
            transport=transport,
            connect_retries=1,
        )

    assert transport.close_count == 0


@pytest.mark.parametrize("client_type", [OpenAIClient, DeepSeekClient, CerebrasClient])
def test_close_is_idempotent_for_owned_injected_transport(client_type):
    transport = CountingTransport(lambda _request: httpx.Response(200))
    client = client_type(api_key=SecretStr("test-key"), transport=transport)

    client.close()
    client.close()

    assert transport.close_count == 1
