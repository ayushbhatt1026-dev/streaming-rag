"""
M4 — builds the initial (version 1) claim graph. This is the structured
upgrade of synthesis/naive_synthesizer.py's plain text: instead of
printable lines, every fact becomes an individually addressable Claim
object the RefinementEngine can patch later without touching the rest.
"""

from __future__ import annotations

import uuid
from typing import Dict, List

from contracts import AnswerVersion, Claim, ClaimStatus, EvidenceChunk, SubQuery


def build_initial_claims(
    sub_queries: List[SubQuery], evidence: List[EvidenceChunk]
) -> AnswerVersion:
    by_sub_query: Dict[str, List[EvidenceChunk]] = {sq.id: [] for sq in sub_queries}
    for chunk in evidence:
        if chunk.sub_query_id in by_sub_query:
            by_sub_query[chunk.sub_query_id].append(chunk)

    claims: List[Claim] = []
    uncertainties: List[str] = []

    for sq in sub_queries:
        chunks = sorted(
            by_sub_query.get(sq.id, []), key=lambda c: -(c.rerank_score or c.rrf_score)
        )
        if not chunks:
            uncertainties.append(f'"{sq.text}" is not documented in the corpus.')
            continue
        best = chunks[0]
        claims.append(
            Claim(
                id=str(uuid.uuid4())[:8],
                text=f'Regarding "{sq.text}": {best.text.strip()}',
                citations=[best.citation()],
                status=ClaimStatus.ADDED,
                introduced_in_version=1,
            )
        )

    return AnswerVersion(version=1, claims=claims, uncertainties=uncertainties)
