"""Retained benchmark for embedding normalization (FIXPLAN §11, PERF-01).

Normalizing one max-batch embedding response is the only place in the library where a
single call touches millions of Python floats, so a redundant full-payload pass here is
not a micro-optimization — it once cost more than the network round-trip it followed.

The timing assertion is calibrated against a reference pass measured on the same machine
in the same run rather than a fixed millisecond budget: absolute budgets either flake on
a loaded CI box or are set so loose they stop catching the regression they exist for.
"""

import time
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from symai.operations import parse_embedding_response
from symai.runtime.models import (
    EmbeddingResponse,
    EmbeddingVector,
    ResponseMetadata,
)

# The documented max-batch case: OpenAI's per-request input ceiling at the largest
# embedding width the library ships a spec for (text-embedding-3-large).
_INPUTS = 2_048
_DIMENSIONS = 3_072

_METADATA = ResponseMetadata(
    provider="openai",
    requested_model="text-embedding-3-large",
    status_code=200,
)


def _payload() -> tuple[float, ...]:
    return tuple(float(index % 7) for index in range(_DIMENSIONS))


def _elapsed[T](operation: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - start


def test_normalizing_a_max_batch_response_costs_less_than_one_redundant_pass() -> None:
    """Normalization must not walk the payload beyond the validation it needs.

    Both historical redundant passes (a pre-scan over every element, and a `float()`
    copy of every value) individually exceeded this bound, so reintroducing either one
    fails here.
    """
    values = _payload()
    raw = tuple((index, values) for index in range(_INPUTS))

    # Reference: exactly one Python-level pass over the whole payload. This is the unit
    # of redundant work the fix removed, and it scales with the machine the test runs on.
    _, reference_s = _elapsed(lambda: [[float(value) for value in row] for _, row in raw])

    response, build_s = _elapsed(
        lambda: EmbeddingResponse(
            vectors=tuple(EmbeddingVector(index=index, values=row) for index, row in raw),
            metadata=_METADATA,
        )
    )
    _, parse_s = _elapsed(lambda: parse_embedding_response(response))

    normalization_s = build_s + parse_s
    assert normalization_s < reference_s * 1.2, (
        f"normalizing {_INPUTS}x{_DIMENSIONS} floats took {normalization_s * 1000:.1f} ms "
        f"vs a {reference_s * 1000:.1f} ms reference pass; a redundant full-payload pass "
        f"has been reintroduced"
    )


def test_parsing_returns_validated_vectors_without_copying_the_payload() -> None:
    """The deterministic half of the benchmark above.

    `EmbeddingVector.values` is already an immutable tuple of finite floats, so parsing
    must hand back those exact objects. This pins the property by identity, independent
    of timing.
    """
    values = _payload()
    response = EmbeddingResponse(
        vectors=(
            EmbeddingVector(index=1, values=values),
            EmbeddingVector(index=0, values=values),
        ),
        metadata=_METADATA,
    )

    parsed = parse_embedding_response(response)

    assert parsed[0] is response.vectors[1].values
    assert parsed[1] is response.vectors[0].values


def test_normalization_still_rejects_non_finite_and_non_numeric_values() -> None:
    """§11 permits no weakening of finite-float validation in exchange for speed."""
    for rejected in (float("inf"), float("nan"), "1.5", True, None):
        with pytest.raises(ValidationError):
            EmbeddingVector(index=0, values=(1.0, rejected))  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValidationError):
        EmbeddingVector(index=0, values=())

    with pytest.raises(ValidationError):
        EmbeddingVector(index=-1, values=(1.0,))
