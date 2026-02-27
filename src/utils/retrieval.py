"""Retrieval helpers for cosine-similarity nearest neighbors."""

from __future__ import annotations

from typing import List

import numpy as np


def topk_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    """Return sorted top-k indices for each query row."""
    if similarities.ndim != 2:
        raise ValueError("Expected 2D similarity matrix [num_queries, num_products].")
    if k <= 0:
        raise ValueError("k must be > 0.")

    num_products = similarities.shape[1]
    k = min(k, num_products)

    # Argpartition is faster than full argsort for large candidate sets.
    partial = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    partial_scores = np.take_along_axis(similarities, partial, axis=1)
    order = np.argsort(-partial_scores, axis=1)
    return np.take_along_axis(partial, order, axis=1)


def retrieve_topk_product_ids(
    query_embeddings: np.ndarray,
    product_embeddings: np.ndarray,
    product_ids: List[str],
    k: int,
) -> List[List[str]]:
    """Retrieve top-k product ids per query using cosine similarity."""
    if query_embeddings.ndim != 2 or product_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if query_embeddings.shape[1] != product_embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match.")
    if len(product_ids) != product_embeddings.shape[0]:
        raise ValueError("product_ids length does not match product embeddings.")

    similarities = query_embeddings @ product_embeddings.T
    idx = topk_indices(similarities, k=k)
    return [[product_ids[i] for i in row] for row in idx]
