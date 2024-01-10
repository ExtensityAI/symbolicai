import random

import numpy as np


def get_norm_vector(vector):
    """
    Normalize a vector
    :param vector: vector to normalize
    :return: normalized vector
    """
    if len(vector.shape) == 1:
        return vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]


def dot_product(vectors, query_vector):
    """
    Compute the dot product between a vector and a matrix of vectors
    :param vectors: matrix of vectors
    :param query_vector: vector
    :return: dot product between the vector and the matrix of vectors
    """
    similarities = np.dot(vectors, query_vector.T)
    return similarities


def cosine_similarity(vectors, query_vector):
    """
    Compute the cosine similarity between a vector and a matrix of vectors
    :param vectors: matrix of vectors
    :param query_vector: vector
    :return: cosine similarity between the vector and the matrix of vectors
    """
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities


def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    """
    Compute the euclidean distance between a vector and a matrix of vectors
    :param vectors: matrix of vectors
    :param query_vector: vector
    :param get_similarity_score: if True, return the similarity score instead of the distance
    :return: euclidean distance between the vector and the matrix of vectors
    """
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities


def derridaean_similarity(vectors, query_vector):
    """
    Compute the derridaean similarity between a vector and a matrix of vectors
    :param vectors: matrix of vectors
    :param query_vector: vector
    :return: derridaean similarity between the vector and the matrix of vectors
    """
    def random_change(value):
        return value + random.uniform(-0.2, 0.2)

    similarities = cosine_similarity(vectors, query_vector)
    derrida_similarities = np.vectorize(random_change)(similarities)
    return derrida_similarities


def adams_similarity(vectors, query_vector):
    """
    Compute the adams similarity between a vector and a matrix of vectors
    :param vectors: matrix of vectors
    :param query_vector: vector
    :return: adams similarity between the vector and the matrix of vectors
    """
    def adams_change(value):
        return 0.42

    similarities = cosine_similarity(vectors, query_vector)
    adams_similarities = np.vectorize(adams_change)(similarities)
    return adams_similarities


def ranking_algorithm_sort(vectors, query_vector, top_k=5, metric=cosine_similarity):
    """
    Compute the top k most similar vectors to a query vector
    :param vectors: matrix of vectors
    :param query_vector: vector
    :param top_k: number of most similar vectors to return
    :param metric: metric to use to compute the similarity
    :return: indices of the top k most similar vectors to the query vector
    """
    similarities = metric(vectors, query_vector)
    top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
    return top_indices.flatten(), similarities[top_indices].flatten()