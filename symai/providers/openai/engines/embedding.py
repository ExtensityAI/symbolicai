from types import MappingProxyType
from typing import cast, override

from pydantic import ValidationError

from symai.providers._client import errors as client_errors
from symai.providers._client.transport import APIResponse
from symai.providers._client.transport import ResponseMetadata as OpenAIResponseMetadata
from symai.providers._engine.base import ProviderEngine, retry_after_seconds
from symai.providers._engine.mapping import ClientErrorMessages, raise_mapped_client_error
from symai.providers.openai.client import Client
from symai.providers.openai.client import embeddings as embeddings_api
from symai.runtime.errors import InvalidResponseError, UnsupportedFeatureError
from symai.runtime.models import (
    EmbeddingModelSpec,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    ProviderId,
    ResponseMetadata,
    TokenUsage,
)

_DIMENSIONALITY_MODELS: frozenset[embeddings_api.Model] = frozenset(
    {"text-embedding-3-small", "text-embedding-3-large"}
)

MODEL_SPECS = MappingProxyType(
    {
        model: EmbeddingModelSpec(dimensions=spec.dimensions)
        for model, spec in embeddings_api.MODEL_SPECS.items()
    }
)

# Shared with the loader, which rejects an unsupported model before allocating transport.
UNSUPPORTED_MODEL_MESSAGE = "Unsupported OpenAI embedding model: {model}"

_ERROR_MESSAGES = ClientErrorMessages(
    authentication="OpenAI rejected authentication",
    rate_limit="OpenAI rate-limited the request",
    response="OpenAI returned an invalid embedding response",
    transport="OpenAI embedding transport failed",
    api="OpenAI embedding request failed with status {status_code}",
)


class EmbeddingEngine(ProviderEngine[Client, embeddings_api.Model, EmbeddingModelSpec]):
    provider: ProviderId = "openai"

    @override
    def __init__(self, *, client: Client, model: str) -> None:
        super().__init__(
            client=client,
            model=model,
            model_specs=MODEL_SPECS,
            unsupported_model_message=UNSUPPORTED_MODEL_MESSAGE,
        )

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
        except client_errors.ClientError as error:
            raise_mapped_client_error(
                error,
                provider=self.provider,
                model=self.model,
                messages=_ERROR_MESSAGES,
            )

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
        response: APIResponse[embeddings_api.EmbeddingList, OpenAIResponseMetadata],
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

        normalized_usage = self._usage(raw.usage)

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
                retry_after=retry_after_seconds(response.metadata.retry_after),
                usage=normalized_usage,
            )
            return EmbeddingResponse(vectors=vectors, metadata=metadata)
        except (TypeError, ValidationError) as error:
            msg = "OpenAI response could not become a normalized embedding response"
            raise InvalidResponseError(msg, metadata=error_metadata) from error

    @staticmethod
    def _usage(usage: embeddings_api.Usage) -> TokenUsage | None:
        if usage.total_tokens != usage.prompt_tokens:
            return None

        try:
            return TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                total_tokens=usage.total_tokens,
            )
        except ValidationError:
            return None
