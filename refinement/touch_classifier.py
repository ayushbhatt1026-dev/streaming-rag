"""
M4 — touch classification: which existing claims does a new constraint
even relate to? Embedding similarity first (cheap, this file). An LLM
contradiction check, if you attach one, runs ONLY on what this flags as
touched — never the full claim set (see refinement_engine.py).

Threshold note: this is a much lower bar than the anti-fragmentation
guard's 0.90 merge threshold (decomposer/anti_fragmentation.py) —
deliberately so. That threshold asks "are these two queries basically the
same question?" This one asks "is this constraint even topically related
to this claim?" — a much looser bar, tuned via config/refinement.yaml.
"""

from __future__ import annotations

from typing import List

from contracts import Claim
from retrieval.embeddings import EmbeddingBackend


def find_touched_claims(
    constraint_text: str,
    claims: List[Claim],
    embedding_backend: EmbeddingBackend,
    threshold: float,
) -> List[Claim]:
    if not claims:
        return []

    texts = [constraint_text] + [c.text for c in claims]
    vectors = embedding_backend.encode(texts)
    constraint_vec = vectors[0]

    touched = []
    for claim, vec in zip(claims, vectors[1:]):
        similarity = float(constraint_vec @ vec)
        if similarity >= threshold:
            touched.append(claim)
    return touched
