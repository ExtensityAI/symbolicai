from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from symai.operations import embedding_request, parse_embedding_response
from symai.symbol import Symbol

if TYPE_CHECKING:
    from symai.runtime.runtime import EmbeddingModel

__all__ = ("embed", "similarity", "distance", "mmd", "kernel")

_MAX_MMD_PAIRWISE_VALUES = 1_000_000


def embed(
    model: EmbeddingModel,
    source: Symbol[str | Sequence[str]],
    *,
    dimensions: int | None = None,
    user: str | None = None,
) -> Symbol[tuple[tuple[float, ...], ...]]:
    inputs = _text_inputs(source)
    response = model.execute(
        embedding_request(inputs, dimensions=dimensions, user=user),
    )
    indices = tuple(vector.index for vector in response.vectors)
    expected_indices = set(range(len(inputs)))
    if len(indices) != len(inputs) or set(indices) != expected_indices:
        msg = "Embedding response indices must exactly match input indices"
        raise ValueError(msg)
    return Symbol(parse_embedding_response(response))


def similarity(
    left: Symbol[Sequence[float] | np.ndarray],
    right: Symbol[Sequence[float] | np.ndarray],
    *,
    metric: Literal["cosine", "dot"] = "cosine",
) -> Symbol[float]:
    lhs, rhs = _matching_vectors(left, right)
    if metric == "dot":
        value = float(np.dot(lhs, rhs))
    elif metric == "cosine":
        left_scale = float(np.max(np.abs(lhs)))
        right_scale = float(np.max(np.abs(rhs)))
        if left_scale == 0 or right_scale == 0:
            msg = "cosine similarity is undefined for a zero vector"
            raise ValueError(msg)
        left_scaled = lhs / left_scale
        right_scaled = rhs / right_scale
        left_normalized = left_scaled / np.linalg.norm(left_scaled)
        right_normalized = right_scaled / np.linalg.norm(right_scaled)
        value = float(np.dot(left_normalized, right_normalized))
    else:
        msg = f"Unsupported similarity metric: {metric!r}"
        raise ValueError(msg)

    return Symbol(value)


def distance(
    left: Symbol[Sequence[float] | np.ndarray],
    right: Symbol[Sequence[float] | np.ndarray],
    *,
    metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
    p: float | None = None,
) -> Symbol[float]:
    lhs, rhs = _matching_vectors(left, right)
    delta = np.abs(lhs - rhs)
    if metric == "euclidean":
        _reject_option(p, "p", metric)
        value = float(np.linalg.norm(delta))
    elif metric == "manhattan":
        _reject_option(p, "p", metric)
        value = float(np.sum(delta))
    elif metric == "minkowski":
        if p is None or not np.isfinite(p) or p < 1:
            msg = "minkowski distance requires an explicit finite p >= 1"
            raise ValueError(msg)
        value = float(np.sum(delta**p) ** (1.0 / p))
    else:
        msg = f"Unsupported distance metric: {metric!r}"
        raise ValueError(msg)

    return Symbol(value)


def mmd(
    left: Symbol[Sequence[Sequence[float]] | np.ndarray],
    right: Symbol[Sequence[Sequence[float]] | np.ndarray],
    *,
    gamma: float,
) -> Symbol[float]:
    _require_positive_finite(gamma, "gamma")
    lhs = _numeric_matrix(left, "left")
    rhs = _numeric_matrix(right, "right")
    if lhs.shape[1] != rhs.shape[1]:
        msg = f"MMD feature shape mismatch: {lhs.shape[1]} != {rhs.shape[1]}"
        raise ValueError(msg)

    sample_count = lhs.shape[0] + rhs.shape[0]
    if sample_count * sample_count > _MAX_MMD_PAIRWISE_VALUES:
        msg = f"MMD pairwise work is bounded to {_MAX_MMD_PAIRWISE_VALUES} values"
        raise ValueError(msg)

    xx = _rbf_matrix(lhs, lhs, gamma)
    yy = _rbf_matrix(rhs, rhs, gamma)
    xy = _rbf_matrix(lhs, rhs, gamma)
    value = float(xx.mean() + yy.mean() - 2.0 * xy.mean())
    return Symbol(value)


def kernel(
    left: Symbol[Sequence[float] | np.ndarray],
    right: Symbol[Sequence[float] | np.ndarray],
    *,
    kind: Literal["linear", "rbf", "polynomial"] = "linear",
    gamma: float | None = None,
    degree: int | None = None,
    coef0: float | None = None,
) -> Symbol[float]:
    lhs, rhs = _matching_vectors(left, right)
    if kind == "linear":
        if gamma is not None or degree is not None or coef0 is not None:
            msg = "gamma, degree, and coef0 are not valid for the linear kernel"
            raise ValueError(msg)
        value = float(np.dot(lhs, rhs))
    elif kind == "rbf":
        if degree is not None or coef0 is not None:
            msg = "degree and coef0 are not valid for the rbf kernel"
            raise ValueError(msg)
        if gamma is None:
            msg = "rbf kernel requires gamma"
            raise ValueError(msg)
        _require_positive_finite(gamma, "gamma")
        value = float(np.exp(-gamma * np.dot(lhs - rhs, lhs - rhs)))
    elif kind == "polynomial":
        if gamma is not None:
            msg = "gamma is not valid for the polynomial kernel"
            raise ValueError(msg)
        if degree is None or coef0 is None:
            msg = "polynomial kernel requires degree and coef0"
            raise ValueError(msg)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            msg = "degree must be a positive integer"
            raise ValueError(msg)
        if not np.isfinite(coef0):
            msg = "coef0 must be finite"
            raise ValueError(msg)
        value = float((np.dot(lhs, rhs) + coef0) ** degree)
    else:
        msg = f"Unsupported kernel kind: {kind!r}"
        raise ValueError(msg)

    return Symbol(value)


def _text_inputs(source: Symbol[str | Sequence[str]]) -> tuple[str, ...]:
    if not isinstance(source, Symbol):
        msg = "source must be a Symbol containing non-empty text input(s)"
        raise TypeError(msg)

    value = source.value
    if isinstance(value, str):
        inputs = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        inputs = tuple(value)
    else:
        msg = "source must be a Symbol containing non-empty text input(s)"
        raise TypeError(msg)

    if not inputs:
        msg = "Embedding requires non-empty text input(s)"
        raise ValueError(msg)
    if any(not isinstance(item, str) or not item for item in inputs):
        msg = "Embedding requires non-empty text input(s)"
        raise TypeError(msg)

    return inputs


def _matching_vectors(
    left: Symbol[Sequence[float] | np.ndarray],
    right: Symbol[Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lhs = _numeric_vector(left, "left")
    rhs = _numeric_vector(right, "right")
    if lhs.shape != rhs.shape:
        msg = f"Vector shape mismatch: {lhs.shape} != {rhs.shape}"
        raise ValueError(msg)

    return lhs, rhs


def _numeric_vector(
    symbol: Symbol[Sequence[float] | np.ndarray],
    field: str,
) -> np.ndarray:
    array = _numeric_array(symbol, field)
    if array.ndim != 1:
        msg = f"{field} must contain a one-dimensional numeric vector"
        raise ValueError(msg)
    if array.size == 0:
        msg = f"{field} must contain a non-empty numeric vector"
        raise ValueError(msg)

    return array


def _numeric_matrix(
    symbol: Symbol[Sequence[Sequence[float]] | np.ndarray],
    field: str,
) -> np.ndarray:
    array = _numeric_array(symbol, field)
    if array.ndim != 2:
        msg = f"{field} must contain a two-dimensional numeric matrix"
        raise ValueError(msg)
    if array.shape[0] == 0 or array.shape[1] == 0:
        msg = f"{field} must contain a non-empty numeric matrix"
        raise ValueError(msg)

    return array


def _numeric_array(symbol: Symbol[object], field: str) -> np.ndarray:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a numeric Symbol"
        raise TypeError(msg)

    try:
        array = np.asarray(symbol.value)
    except (TypeError, ValueError) as error:
        msg = f"{field} must contain regular numeric values"
        raise TypeError(msg) from error

    if array.dtype.kind not in "iuf":
        msg = f"{field} must contain real numeric values"
        raise TypeError(msg)
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        msg = f"{field} must contain only finite numeric values"
        raise ValueError(msg)

    return array


def _rbf_matrix(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    np.maximum(distances, 0.0, out=distances)
    return np.exp(-gamma * distances)


def _require_positive_finite(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        msg = f"{field} must be a positive finite number"
        raise TypeError(msg)
    if not np.isfinite(value) or value <= 0:
        msg = f"{field} must be a positive finite number"
        raise ValueError(msg)


def _reject_option(value: object, option: str, metric: str) -> None:
    if value is not None:
        msg = f"{option} is only valid for minkowski distance, not {metric}"
        raise ValueError(msg)
