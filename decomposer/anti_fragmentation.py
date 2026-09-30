"""
M2 — anti-fragmentation guard (blueprint Step 6, guide pitfall #5).

Two rules, applied in order:
  1. Merge: any pair of candidate sub-queries with cosine similarity above
     the threshold get collapsed to one (the first one seen).
  2. Cap: if more than max_sub_queries survive, keep the most DISTINCT ones
     — not just the first N — by dropping whichever candidate is, on
     average, most similar to the others.

Reuses whatever embedding backend the rest of the system is using (BGE in
production, TF-IDF offline) — see retrieval/embeddings.py. Passing the same
backend in means "distinct" here means the same thing it means in M3.
"""

from __future__ import annotations

from typing import List

import numpy as np

from retrieval.embeddings import EmbeddingBackend


def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    # vectors are L2-normalized by the embedding backends, so dot product
    # == cosine similarity.
    return vectors @ vectors.T


def merge_near_duplicates(
    candidates: List[dict], embedding_backend: EmbeddingBackend, threshold: float,
) -> List[dict]:
    """Each candidate is {"text": ..., "intent_type": ...}. Keeps the first
    of any pair whose texts are near-duplicate."""
    if len(candidates) <= 1:
        return candidates

    texts = [c["text"] for c in candidates]
    vectors = embedding_backend.encode(texts)
    sims = _cosine_sim_matrix(vectors)

    kept_idx: List[int] = []
    for i in range(len(candidates)):
        is_dup = any(sims[i, j] >= threshold for j in kept_idx)
        if not is_dup:
            kept_idx.append(i)
    return [candidates[i] for i in kept_idx]


def cap_by_distinctness(
    candidates: List[dict], embedding_backend: EmbeddingBackend, max_k: int,
) -> List[dict]:
    """If more than max_k candidates survive merging, keep the max_k most
    mutually distinct ones — measured as lowest average similarity to the
    rest of the set — rather than an arbitrary first-N truncation."""
    if len(candidates) <= max_k:
        return candidates

    texts = [c["text"] for c in candidates]
    vectors = embedding_backend.encode(texts)
    sims = _cosine_sim_matrix(vectors)
    np.fill_diagonal(sims, 0.0)
    avg_similarity = sims.mean(axis=1)

    # Lower avg similarity to the rest = more distinct = keep first.
    order = np.argsort(avg_similarity)[:max_k]
    order = sorted(order.tolist())  # preserve original relative order
    return [candidates[i] for i in order]
