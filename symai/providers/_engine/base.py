from collections.abc import Mapping
from math import isfinite
from typing import Never, Protocol, cast

from symai.providers._client.transport import ResponseMetadata
from symai.runtime.errors import (
    ErrorMetadata,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import ProviderId


class _Closeable(Protocol):
    def close(self) -> None: ...


def retry_after_seconds(value: float | None) -> float | None:
    """Keep only a finite, non-negative retry delay; treat any other value as absent."""
    return value if value is not None and value >= 0 and isfinite(value) else None


class ProviderEngine[ClientT: _Closeable, ModelT: str, ModelSpecT]:
    # Set by each engine. Declared here because `_error_metadata` stamps it onto every
    # error the engine raises.
    provider: ProviderId

    def __init__(
        self,
        *,
        client: ClientT,
        model: str,
        model_specs: Mapping[str, ModelSpecT],
        unsupported_model_message: str,
    ) -> None:
        try:
            try:
                model_spec = model_specs[model]
            except KeyError as error:
                msg = unsupported_model_message.format(model=model)
                raise UnsupportedModelError(msg) from error

            self._client = client
            self._model = cast("ModelT", model)
            self._model_spec = model_spec
            self._closed = False
        except BaseException as error:
            try:
                client.close()
            except BaseException as cleanup_error:
                error.add_note(f"Engine construction cleanup failed: {cleanup_error!r}")
            raise

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._client.close()

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def model_spec(self) -> ModelSpecT:
        return self._model_spec

    def _error_metadata(self, metadata: ResponseMetadata) -> ErrorMetadata:
        """Identify the failing engine without disclosing anything the provider sent.

        Every provider derives this the same way, and a provider that specializes its
        response metadata (Cerebras adds rate-limit state) subclasses the shared type, so
        it is accepted here too.
        """
        return ErrorMetadata(
            provider=self.provider,
            model=self.model,
            request_id=metadata.request_id,
            retry_after=retry_after_seconds(metadata.retry_after),
        )

    @staticmethod
    def _unsupported(message: str) -> Never:
        raise UnsupportedFeatureError(message)
