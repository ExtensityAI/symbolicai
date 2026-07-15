from math import isfinite
from types import MappingProxyType
from typing import cast

from pydantic import ValidationError

from symai.providers.openai.client import embeddings as embeddings_api
from symai.providers.openai.client import errors as openai_errors
from symai.providers.openai.client import Client
from symai.providers.openai.client.transport import APIResponse
from symai.providers.openai.client.transport import ResponseMetadata as OpenAIResponseMetadata
from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import (
    EmbeddingModelSpec,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    Provider,
    ResponseMetadata,
    TokenUsage,
)

_DIMENSIONALITY_MODELS: frozenset[embeddings_api.Model] = frozenset(
    {"text-embedding-3-small", "text-embedding-3-large"}
)

MODEL_SPECS = MappingProxyType(
    {
        model: EmbeddingModelSpec(
            context_tokens=spec.context_tokens,
            dimensions=spec.dimensions,
        )
        for model, spec in embeddings_api.MODEL_SPECS.items()
    }
)


class EmbeddingEngine:
    provider = Provider.OPENAI

    def __init__(self, *, client: Client, model: str) -> None:
        try:
            model_spec = MODEL_SPECS[model]
        except KeyError as error:
            msg = f"Unsupported OpenAI embedding model: {model}"
            raise UnsupportedModelError(msg) from error

        self._client = client
        self._model: embeddings_api.Model = cast("embeddings_api.Model", model)
        self._model_spec = model_spec

    @property
    def model(self) -> embeddings_api.Model:
        return self._model

    @property
    def model_spec(self) -> EmbeddingModelSpec:
        return self._model_spec

    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._validate_request(request)
        provider_request = embeddings_api.CreateEmbeddingRequest(
            input=request.inputs,
            model=self.model,
            dimensions=request.dimensions,
            encoding_format="float",
            user=request.user,
        )
        try:
            response = self._client.create_embeddings(provider_request)
        except openai_errors.AuthError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI rejected authentication"
            raise AuthenticationError(msg, metadata=metadata) from error
        except openai_errors.RateLimitError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI rate-limited the request"
            raise RateLimitError(msg, metadata=metadata) from error
        except openai_errors.ResponseError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "OpenAI returned an invalid embedding response"
            raise InvalidResponseError(msg, metadata=metadata) from error
        except openai_errors.TransportError as error:
            metadata = ErrorMetadata(provider=self.provider, model=self.model)
            msg = "OpenAI embedding transport failed"
            raise TransportError(msg, metadata=metadata) from error
        except openai_errors.APIError as error:
            metadata = self._error_metadata(error.metadata)
            msg = f"OpenAI embedding request failed with status {error.metadata.status_code}"
            raise ExecutionError(msg, metadata=metadata) from error

        return self._parse_response(
            response,
            expected_count=len(request.inputs),
            expected_dimensions=request.dimensions or self.model_spec.dimensions,
        )

    def _validate_request(self, request: EmbeddingRequest) -> None:
        dimensions = request.dimensions
        if dimensions is None:
            return
        if self.model not in _DIMENSIONALITY_MODELS:
            msg = f"OpenAI embedding model {self.model} does not support dimensions"
            raise UnsupportedFeatureError(msg)
        if dimensions > self.model_spec.dimensions:
            msg = (
                f"OpenAI embedding model {self.model} supports dimensions up to "
                f"{self.model_spec.dimensions}"
            )
            raise UnsupportedFeatureError(msg)

    def _parse_response(
        self,
        response: APIResponse[embeddings_api.EmbeddingList],
        *,
        expected_count: int,
        expected_dimensions: int,
    ) -> EmbeddingResponse:
        raw = response.data
        error_metadata = self._error_metadata(response.metadata)

        seen_indices: set[int] = set()
        for item in raw.data:
            if item.index in seen_indices:
                msg = "OpenAI embedding response contained duplicate indices"
                raise InvalidResponseError(msg, metadata=error_metadata)
            seen_indices.add(item.index)
            if isinstance(item.embedding, str):
                msg = "OpenAI embedding response did not contain float vectors"
                raise InvalidResponseError(msg, metadata=error_metadata)
            if len(item.embedding) != expected_dimensions:
                msg = "OpenAI embedding response contained unexpected dimensions"
                raise InvalidResponseError(msg, metadata=error_metadata)

        ordered = sorted(raw.data, key=lambda item: item.index)
        if len(ordered) != expected_count or any(
            item.index != expected_index for expected_index, item in enumerate(ordered)
        ):
            msg = "OpenAI embedding response indices did not match the request"
            raise InvalidResponseError(msg, metadata=error_metadata)

        usage = raw.usage
        if usage.total_tokens != usage.prompt_tokens:
            msg = "OpenAI embedding token usage was inconsistent"
            raise InvalidResponseError(msg, metadata=error_metadata)

        try:
            vectors = tuple(
                EmbeddingVector(index=item.index, values=cast("tuple[float, ...]", item.embedding))
                for item in ordered
            )
            metadata = ResponseMetadata(
                provider=self.provider,
                requested_model=self.model,
                response_model=raw.model,
                status_code=response.metadata.status_code,
                request_id=response.metadata.request_id,
                retry_after=self._retry_after(response.metadata.retry_after),
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
            return EmbeddingResponse(vectors=vectors, metadata=metadata)
        except (TypeError, ValidationError) as error:
            msg = "OpenAI response could not become a normalized embedding response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    def _error_metadata(
        self,
        metadata: OpenAIResponseMetadata | None,
    ) -> ErrorMetadata:
        return ErrorMetadata(
            provider=self.provider,
            model=self.model,
            request_id=metadata.request_id if metadata is not None else None,
            retry_after=self._retry_after(metadata.retry_after if metadata is not None else None),
        )

    @staticmethod
    def _retry_after(value: float | None) -> float | None:
        return value if value is not None and value >= 0 and isfinite(value) else None
