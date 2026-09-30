"""
Naive synthesizer — Phase 1 Step 4's placeholder, per the blueprint:
"chunks in, cited answer out, no state." Deliberately NOT the real M5
(Grounded Synthesizer with claim graphs, entailment verification, LLM
generation) — that's a later step you haven't asked for yet.

This version is extractive and deterministic (no LLM call at all): for each
sub-query, take its best-scoring evidence chunk and quote a short excerpt
with its citation. If a sub-query has no evidence, say so explicitly rather
than inventing an answer — this is a cheap head start on the "explicit
uncertainty" behaviour real M5 will formalize later.

Being deterministic and LLM-free means this is fully testable offline,
which is exactly what you want while wiring up the walking skeleton.
"""

from __future__ import annotations

from typing import Dict, List

from contracts import EvidenceChunk, SubQuery


def _excerpt(text: str, max_chars: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


class NaiveSynthesizer:
    def synthesize(
        self, sub_queries: List[SubQuery], evidence: List[EvidenceChunk]
    ) -> str:
        by_sub_query: Dict[str, List[EvidenceChunk]] = {sq.id: [] for sq in sub_queries}
        for chunk in evidence:
            if chunk.sub_query_id in by_sub_query:
                by_sub_query[chunk.sub_query_id].append(chunk)

        lines: List[str] = []
        for sq in sub_queries:
            chunks = sorted(
                by_sub_query.get(sq.id, []),
                key=lambda c: -(c.rerank_score or c.rrf_score),
            )
            if not chunks:
                lines.append(f"- Re: \"{sq.text}\" — not documented in the corpus.")
                continue
            best = chunks[0]
            # Whitelist check, in spirit: only cite a chunk we actually
            # retrieved for THIS sub-query — trivially true here since we
            # only ever read from `evidence`, but stated explicitly because
            # this is exactly the rule the real M5 whitelist filter
            # enforces mechanically later, against LLM-generated text.
            lines.append(f"- Re: \"{sq.text}\" — {_excerpt(best.text)} {best.citation()}")

        return "\n".join(lines)
