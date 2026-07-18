"""Shared mock harness for engine tests: routes an engine's transport through
httpx.MockTransport and records requests. Category contracts build on this:
tests/engines/neurosymbolic/interface.py and tests/engines/search/interface.py.
"""

import json

import httpx

DUMMY_KEY = "sk-test-not-a-real-key"


class MockAPI:
    """Routes an engine's transport through httpx.MockTransport and records requests."""

    def __init__(self, engine, handler):
        def spy(request):
            self.requests.append(request)
            return handler(request)

        self.requests = []
        self.client = httpx.Client(transport=httpx.MockTransport(spy))
        engine.transport_client = self.client

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.client.close()

    @property
    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content.decode("utf-8"))
