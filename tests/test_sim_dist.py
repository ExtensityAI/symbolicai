import random

import numpy as np
import pytest

from symai import Symbol
from symai.extended.metrics.similarity import (
    adams_similarity,
    cosine_similarity as legacy_cosine_similarity,
    derridaean_similarity,
    dot_product,
    euclidean_metric,
    get_norm_vector,
    ranking_algorithm_sort,
)
from symai.ops.measures import calculate_frechet_distance, calculate_mmd

BASE_VECTOR = np.array([1.0, 3.0, 5.0], dtype=float)
OTHER_VECTOR = np.array([2.0, 4.0, 6.0], dtype=float)
EPSILON = 1e-8
SIGMA1 = np.array(
    [
        [1.0, 0.1, 0.0],
        [0.1, 1.2, 0.1],
        [0.0, 0.1, 1.1],
    ],
    dtype=float,
)
SIGMA2 = np.array(
    [
        [1.3, 0.0, 0.2],
        [0.0, 1.1, 0.0],
        [0.2, 0.0, 1.4],
    ],
    dtype=float,
)
RBF_BANDWIDTH = (0.5, 1.5)


def _dot(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.dot(lhs, rhs))


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _squared_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    diff = lhs - rhs
    return float(np.sum(diff * diff))


def _abs_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sum(np.abs(lhs - rhs)))


def _expected_similarity_cosine(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    numerator = _dot(lhs, rhs)
    denominator = _norm(lhs) * _norm(rhs) + eps
    return numerator / denominator


def _expected_similarity_angular(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    cosine = _expected_similarity_cosine(lhs, rhs, eps, {})
    ratio = np.clip(cosine, -1.0, 1.0)
    c = kwargs.get('c', 1)
    return 1 - (c * np.arccos(ratio) / np.pi)


def _expected_similarity_product(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return _dot(lhs, rhs)


def _expected_similarity_manhattan(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return _abs_distance(lhs, rhs)


def _expected_similarity_euclidean(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return float(np.linalg.norm(lhs - rhs))


def _expected_similarity_minkowski(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    p = kwargs.get('p', 3)
    if p <= 0:
        raise ValueError('Minkowski order must be positive')
    return float(np.sum(np.abs(lhs - rhs) ** p) ** (1.0 / p))


def _expected_similarity_jaccard(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    intersection = np.minimum(lhs, rhs)
    union = np.maximum(lhs, rhs)
    return float(np.sum(intersection) / (np.sum(union) + eps))


SIMILARITY_EXPECTED = {
    'cosine': _expected_similarity_cosine,
    'angular-cosine': _expected_similarity_angular,
    'product': _expected_similarity_product,
    'manhattan': _expected_similarity_manhattan,
    'euclidean': _expected_similarity_euclidean,
    'minkowski': _expected_similarity_minkowski,
    'jaccard': _expected_similarity_jaccard,
}

SIMILARITY_KWARGS = {metric: {} for metric in SIMILARITY_EXPECTED}


def _expected_kernel_gaussian(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    return float(np.exp(-gamma * _squared_distance(lhs, rhs)))


def _expected_kernel_rbf(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    distance_sq = _squared_distance(lhs, rhs)
    bandwidth = kwargs.get('bandwidth')
    if bandwidth is not None:
        total = 0.0
        for width in bandwidth:
            gamma = 1.0 / (2 * width)
            total += np.exp(-gamma * distance_sq)
        return float(total)
    gamma = kwargs.get('gamma', 1)
    return float(np.exp(-gamma * distance_sq))


def _expected_kernel_laplacian(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    return float(np.exp(-gamma * _abs_distance(lhs, rhs)))


def _expected_kernel_polynomial(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    degree = kwargs.get('degree', 3)
    coef = kwargs.get('coef', 1)
    return float((gamma * _dot(lhs, rhs) + coef) ** degree)


def _expected_kernel_sigmoid(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    coef = kwargs.get('coef', 1)
    return float(np.tanh(gamma * _dot(lhs, rhs) + coef))


def _expected_kernel_linear(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return _dot(lhs, rhs)


def _expected_kernel_cauchy(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    return float(1 / (1 + _squared_distance(lhs, rhs) / gamma))


def _expected_kernel_t_distribution(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    degree = kwargs.get('degree', 1)
    base = _squared_distance(lhs, rhs) / (gamma * degree)
    value = (base ** (degree + 1)) / 2
    return float(1 / (1 + value))


def _expected_kernel_inverse_multiquadric(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    gamma = kwargs.get('gamma', 1)
    return float(1 / np.sqrt(_squared_distance(lhs, rhs) / (gamma ** 2) + 1))


def _expected_kernel_cosine(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return 1 - _expected_similarity_cosine(lhs, rhs, eps, {})


def _expected_kernel_angular_cosine(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    cosine = _expected_similarity_cosine(lhs, rhs, eps, {})
    ratio = np.clip(cosine, -1.0, 1.0)
    c = kwargs.get('c', 1)
    return float(c * np.arccos(ratio) / np.pi)


def _expected_kernel_frechet(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return float(calculate_frechet_distance(lhs, kwargs['sigma1'], rhs, kwargs['sigma2'], eps=eps))


def _expected_kernel_mmd(lhs: np.ndarray, rhs: np.ndarray, eps: float, kwargs: dict) -> float:
    return float(calculate_mmd(lhs[None, :], rhs[None, :], eps=eps))


KERNEL_EXPECTED = {
    'gaussian': _expected_kernel_gaussian,
    'rbf': _expected_kernel_rbf,
    'laplacian': _expected_kernel_laplacian,
    'polynomial': _expected_kernel_polynomial,
    'sigmoid': _expected_kernel_sigmoid,
    'linear': _expected_kernel_linear,
    'cauchy': _expected_kernel_cauchy,
    't-distribution': _expected_kernel_t_distribution,
    'inverse-multiquadric': _expected_kernel_inverse_multiquadric,
    'cosine': _expected_kernel_cosine,
    'angular-cosine': _expected_kernel_angular_cosine,
    'frechet': _expected_kernel_frechet,
    'mmd': _expected_kernel_mmd,
}

KERNEL_KWARGS = {
    'gaussian': {},
    'rbf': {'bandwidth': RBF_BANDWIDTH},
    'laplacian': {},
    'polynomial': {},
    'sigmoid': {},
    'linear': {},
    'cauchy': {},
    't-distribution': {},
    'inverse-multiquadric': {},
    'cosine': {},
    'angular-cosine': {},
    'frechet': {'sigma1': SIGMA1, 'sigma2': SIGMA2},
    'mmd': {},
}


@pytest.mark.parametrize('metric', SIMILARITY_EXPECTED.keys())
def test_symbol_similarity_metrics(metric: str) -> None:
    base_symbol = Symbol(BASE_VECTOR.tolist())
    kwargs = dict(SIMILARITY_KWARGS.get(metric, {}))
    result = base_symbol.similarity(OTHER_VECTOR.tolist(), metric=metric, eps=EPSILON, **kwargs)
    expected = SIMILARITY_EXPECTED[metric](BASE_VECTOR, OTHER_VECTOR, EPSILON, kwargs)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize('kernel', KERNEL_EXPECTED.keys())
def test_symbol_distance_kernels(kernel: str) -> None:
    base_symbol = Symbol(BASE_VECTOR.tolist())
    kwargs = dict(KERNEL_KWARGS.get(kernel, {}))
    result = base_symbol.distance(OTHER_VECTOR.tolist(), kernel=kernel, eps=EPSILON, **kwargs)
    expected = KERNEL_EXPECTED[kernel](BASE_VECTOR, OTHER_VECTOR, EPSILON, kwargs)
    assert result == pytest.approx(expected)


def test_get_norm_vector_normalizes_vectors() -> None:
    vector = np.array([3.0, 4.0], dtype=float)
    normalized = get_norm_vector(vector)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)

    matrix = np.array([[3.0, 4.0], [0.0, 3.0]], dtype=float)
    normalized_matrix = get_norm_vector(matrix)
    norms = np.linalg.norm(normalized_matrix, axis=1)
    assert norms == pytest.approx(np.ones_like(norms, dtype=float))


def test_dot_and_cosine_similarity_helpers() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    query = np.array([1.0, 2.0, 3.0], dtype=float)
    expected_dot = vectors @ query
    assert dot_product(vectors, query) == pytest.approx(expected_dot)

    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    norm_query = query / np.linalg.norm(query)
    expected_cosine = norm_vectors @ norm_query
    assert legacy_cosine_similarity(vectors, query) == pytest.approx(expected_cosine)


def test_euclidean_metric_distance_toggle() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    query = np.array([1.0, 2.0, 3.0], dtype=float)
    distances = euclidean_metric(vectors, query, get_similarity_score=False)
    expected_distances = np.linalg.norm(vectors - query, axis=1)
    assert distances == pytest.approx(expected_distances)

    similarities = euclidean_metric(vectors, query)
    expected_similarities = 1 / (1 + expected_distances)
    assert similarities == pytest.approx(expected_similarities)


def test_derridaean_similarity_seeded_noise() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    query = np.array([1.0, 2.0, 3.0], dtype=float)
    random.seed(42)
    expected = derridaean_similarity(vectors, query)
    random.seed(42)
    result = derridaean_similarity(vectors, query)

    base_cosine = legacy_cosine_similarity(vectors, query)
    deltas = expected - base_cosine

    assert result == pytest.approx(expected)
    assert np.all(deltas >= -0.2)
    assert np.all(deltas <= 0.2)


def test_adams_similarity_returns_constant() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    query = np.array([1.0, 2.0, 3.0], dtype=float)
    result = adams_similarity(vectors, query)
    expected = np.full_like(result, 0.42, dtype=float)
    assert result == pytest.approx(expected)


def test_ranking_algorithm_sort_orders_results() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    query = np.array([1.0, 1.0, 0.0], dtype=float)
    indices, scores = ranking_algorithm_sort(vectors, query, top_k=3)

    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    norm_query = query / np.linalg.norm(query)
    similarities = norm_vectors @ norm_query
    expected_indices = np.argsort(similarities)[-3:][::-1]
    expected_scores = similarities[expected_indices]

    assert indices.tolist() == expected_indices.tolist()
    assert scores == pytest.approx(expected_scores)
