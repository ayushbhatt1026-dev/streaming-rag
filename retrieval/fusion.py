"""
M3 — Reciprocal Rank Fusion, dedupe, and reranking.

Why RRF (per blueprint §2, M3): score-scale-free, no tuning of dense/sparse
weights, robust, one function. It just needs each ranked list's *order*,
not comparable scores — which matters because a BM25 score and a cosine
similarity score live on completely different scales.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

import numpy as np

from contracts import EvidenceChunk

RRF_K = 60  # standard constant from the RRF literature; rarely needs tuning
DEDUPE_SIMILARITY_THRESHOLD = 0.95


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]], k: int = RRF_K
) -> Dict[str, float]:
    """Each input is a ranked list of (chunk_id, raw_score) — raw_score is
    ignored, only rank/order matters. Returns {chunk_id: rrf_score}."""
    fused: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def dedupe_near_duplicates(
    chunks: List[EvidenceChunk], vectors: Dict[str, np.ndarray],
    threshold: float = DEDUPE_SIMILARITY_THRESHOLD,
) -> List[EvidenceChunk]:
    """Exact-hash dedupe first (identical text), then near-duplicate merge
    by embedding cosine similarity. Keeps the higher-rrf_score chunk of any
    duplicate pair."""
    # Exact-hash pass
    seen_text: Dict[str, EvidenceChunk] = {}
    for c in chunks:
        key = c.text.strip()
        if key not in seen_text or c.rrf_score > seen_text[key].rrf_score:
            seen_text[key] = c
    deduped = list(seen_text.values())

    # Near-duplicate pass (only meaningful once vectors are available)
    kept: List[EvidenceChunk] = []
    for c in sorted(deduped, key=lambda x: -x.rrf_score):
        c_vec = vectors.get(c.chunk_id)
        is_dup = False
        if c_vec is not None:
            for k in kept:
                k_vec = vectors.get(k.chunk_id)
                if k_vec is None:
                    continue
                sim = float(np.dot(c_vec, k_vec))
                if sim >= threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
    return kept


class Reranker(Protocol):
    def rerank(self, query: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]: ...


class NoOpReranker:
    """Keeps RRF order. This is the default until you plug in a real
    cross-encoder (bge-reranker-v2-m3 or ms-marco-MiniLM-L-6-v2 per the
    tooling table) — both also require a one-time model download, so this
    stub keeps the pipeline runnable offline in the meantime."""

    def rerank(self, query: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        for c in chunks:
            c.rerank_score = c.rrf_score
        return sorted(chunks, key=lambda c: -c.rrf_score)
