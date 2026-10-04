"""
M4 — RefinementEngine. Given a session's existing AnswerVersion and a new
constraint (the text of a REFINE-classified turn), patches only the
affected claims and issues only targeted delta sub-queries — never a full
corpus re-search.

The "no full-corpus query during REFINE" guarantee (what G5 actually
checks, per the guide) is structural here, not a separate rule to
remember: `decomposer.decompose()` is called with ONLY `constraint_text`,
never the original first-turn utterance. There is no code path in this
class that has access to the original utterance at all — see
tests/test_refinement.py for an automated check of exactly this.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from contracts import AnswerVersion, Claim, ClaimStatus, TelemetryEvent
from decomposer.decomposer import MultiIntentDecomposer
from refinement.touch_classifier import find_touched_claims
from retrieval.embeddings import EmbeddingBackend
from retrieval.retriever import Retriever
from telemetry.bus import TelemetryBus


def _load_config(path: str = "config/refinement.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RefinementEngine:
    decomposer: MultiIntentDecomposer
    retriever: Retriever
    embedding_backend: EmbeddingBackend
    telemetry: TelemetryBus
    session_id: str
    trace_id: str
    config_path: str = "config/refinement.yaml"
    contradiction_checker: Optional[object] = None  # see contradiction_checker.py

    def __post_init__(self) -> None:
        cfg = _load_config(self.config_path)
        self._touch_threshold = cfg["touch_similarity_threshold"]

    def refine(self, prior: AnswerVersion, constraint_text: str) -> AnswerVersion:
        new_version_num = prior.version + 1

        touched = find_touched_claims(
            constraint_text, prior.claims, self.embedding_backend, self._touch_threshold
        )
        touched_ids = {c.id for c in touched}

        # Targeted delta retrieval. This call sees ONLY constraint_text —
        # that's what makes "no full-corpus query" structurally true rather
        # than a convention someone could forget.
        delta_sub_queries = self.decomposer.decompose(
            constraint_text, parent_utterance_id=self.trace_id
        )
        delta_evidence = self.retriever.search(delta_sub_queries)

        new_claims: List[Claim] = []
        changed_claim_ids: List[str] = []

        # Carry every prior claim forward first. Untouched claims are
        # copied verbatim with their ORIGINAL citations -- this IS the
        # "don't restart" guarantee, not just a design intention.
        for claim in prior.claims:
            if claim.id in touched_ids and self.contradiction_checker is not None:
                contradicted = self.contradiction_checker.check(
                    claim, constraint_text, delta_evidence
                )
                if contradicted:
                    new_claims.append(
                        Claim(
                            id=claim.id,
                            text=claim.text,
                            citations=claim.citations,
                            status=ClaimStatus.RETRACTED,
                            introduced_in_version=claim.introduced_in_version,
                        )
                    )
                    changed_claim_ids.append(claim.id)
                    continue
            # Untouched (or touched-but-not-contradicted) -> KEPT in THIS
            # version. Status describes what happened to the claim in the
            # version being built, not frozen at whatever it was when
            # first introduced -- a claim is only ever ADDED in the one
            # version it first appears, every version after that it's
            # either KEPT, AMENDED, or RETRACTED.
            new_claims.append(
                Claim(
                    id=claim.id,
                    text=claim.text,
                    citations=claim.citations,
                    status=ClaimStatus.KEPT,
                    introduced_in_version=claim.introduced_in_version,
                    superseded_by=claim.superseded_by,
                )
            )

        # Add new claims from the delta evidence.
        by_sub_query: Dict[str, list] = {sq.id: [] for sq in delta_sub_queries}
        for chunk in delta_evidence:
            if chunk.sub_query_id in by_sub_query:
                by_sub_query[chunk.sub_query_id].append(chunk)

        new_uncertainties = list(prior.uncertainties)
        for sq in delta_sub_queries:
            chunks = sorted(
                by_sub_query.get(sq.id, []), key=lambda c: -(c.rerank_score or c.rrf_score)
            )
            if not chunks:
                msg = f'"{sq.text}" is not documented in the corpus.'
                if msg not in new_uncertainties:
                    new_uncertainties.append(msg)
                continue
            best = chunks[0]
            new_claim = Claim(
                id=str(uuid.uuid4())[:8],
                text=f'Regarding "{sq.text}": {best.text.strip()}',
                citations=[best.citation()],
                status=ClaimStatus.ADDED,
                introduced_in_version=new_version_num,
            )
            new_claims.append(new_claim)
            changed_claim_ids.append(new_claim.id)

        new_version = AnswerVersion(
            version=new_version_num, claims=new_claims, uncertainties=new_uncertainties
        )

        self.telemetry.emit(
            TelemetryEvent(
                trace_id=self.trace_id,
                session_id=self.session_id,
                stage="refinement",
                event="version_transition",
                t_ms=0.0,
                payload={
                    "from_version": prior.version,
                    "to_version": new_version_num,
                    "touched_claim_ids": list(touched_ids),
                    "changed_claim_ids": changed_claim_ids,
                    "delta_sub_queries": [sq.text for sq in delta_sub_queries],
                },
            )
        )
        return new_version
