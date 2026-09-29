"""
Phase 0, Step 2 — the seven frozen interface contracts.

A module may only see its own input contract and produce its own output
contract. No reaching into another module's internals. This file is the
ONLY thing every module is allowed to import from every other module.

Implementation note on library choice: the tooling table's recommended
default is Pydantic v2 (typed validation + free LLM structured-output
enforcement). This file uses Python's built-in `dataclasses` instead,
because Pydantic isn't installable in the environment these contracts were
authored and tested in (no internet access to `pip install`), and every
class below has actually been imported, instantiated, and exercised against
the real retrieval module (see the smoke test at the bottom, and
`demo_retrieval.py`). Swapping to Pydantic later is a mechanical,
low-risk change — see the "Pydantic v2 upgrade path" note after the
classes — but don't do it without re-running that smoke test, since
Pydantic's validation is stricter than a plain dataclass's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. TranscriptChunk — output of the client / simulator, input to StreamBuffer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscriptChunk:
    """One incremental slice of a still-forming spoken utterance, as it
    arrives over the WebSocket every ~200-400ms."""

    session_id: str
    seq: int                  # monotonically increasing per session
    t_ms: float                # milliseconds since utterance start
    text_delta: str            # the new text since the last chunk
    is_final: bool = False     # True on the chunk that ends the utterance


# ---------------------------------------------------------------------------
# 2. ControllerDecision — output of M1 (Retrieval Controller)
# ---------------------------------------------------------------------------
class Decision(str, Enum):
    """The five-way decision taxonomy. Fix this early — telemetry and every
    downstream module branch on these exact values."""

    WAIT = "WAIT"
    RETRIEVE_PROVISIONAL = "RETRIEVE_PROVISIONAL"
    RETRIEVE_FINAL = "RETRIEVE_FINAL"
    REFINE = "REFINE"
    SUPPRESS = "SUPPRESS"


@dataclass
class ControllerDecision:
    """What M1 hands to everything downstream. `sub_intent_hint` is an
    optional early guess at how many distinct intents are present — M2 is
    free to ignore it and decide for itself."""

    decision: Decision
    confidence: float                       # 0.0-1.0
    reason: str                             # short, human-readable, for telemetry/debugging
    sub_intent_hint: List[str] = field(default_factory=list)
    is_refinement_of_prior_answer: bool = False


# ---------------------------------------------------------------------------
# 3. SubQuery — output of M2 (Multi-Intent Decomposer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubQuery:
    """One independently answerable, search-ready question. Self-contained:
    pronouns resolved, session topic carried forward."""

    id: str
    text: str
    intent_type: str = "factual_lookup"  # factual_lookup | policy | constraint | comparison
    parent_utterance_id: str = ""


# ---------------------------------------------------------------------------
# 4. EvidenceChunk — output of M3 (Retrieval & Evidence Fusion)
# ---------------------------------------------------------------------------
@dataclass
class EvidenceChunk:
    """A single retrievable unit of the corpus, plus its retrieval scores
    for this turn. Citations downstream are only as good as doc_id +
    section_ref, so never let these be empty."""

    chunk_id: str
    doc_id: str
    section_ref: str
    heading_path: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    sub_query_id: str = ""

    def citation(self) -> str:
        """The exact label M5 (the Synthesizer) is allowed to cite."""
        return f"[{self.doc_id} §{self.section_ref}]"


# ---------------------------------------------------------------------------
# 5. Claim — the atomic unit inside an AnswerVersion (M4 / M5)
# ---------------------------------------------------------------------------
class ClaimStatus(str, Enum):
    KEPT = "KEPT"              # carried forward unchanged from a prior version
    AMENDED = "AMENDED"        # existed before, text/citations changed this version
    ADDED = "ADDED"            # new in this version
    RETRACTED = "RETRACTED"    # existed before, no longer asserted


@dataclass
class Claim:
    """One factual sentence, individually tagged so the Refinement Engine
    (M4) can patch just this claim instead of the whole answer."""

    id: str
    text: str
    citations: List[str] = field(default_factory=list)  # e.g. ["[Doc_12 §2]"]
    status: ClaimStatus = ClaimStatus.ADDED
    introduced_in_version: int = 1
    superseded_by: Optional[str] = None  # claim id that replaced this one, if any


# ---------------------------------------------------------------------------
# 6. AnswerVersion — output of M4/M5, the versioned claim graph
# ---------------------------------------------------------------------------
@dataclass
class AnswerVersion:
    """The full answer state for a session at a point in time. Refinement
    increments `version` and mutates only the touched claims — untouched
    claims are copied forward with their original citations."""

    version: int
    claims: List[Claim] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

    @property
    def citations_all(self) -> List[str]:
        """Derived, not stored redundantly: the flat citation list across
        every non-retracted claim."""
        seen: List[str] = []
        for c in self.claims:
            if c.status == ClaimStatus.RETRACTED:
                continue
            for cite in c.citations:
                if cite not in seen:
                    seen.append(cite)
        return seen


# ---------------------------------------------------------------------------
# 7. TelemetryEvent — emitted by every module, consumed by M6
# ---------------------------------------------------------------------------
@dataclass
class TelemetryEvent:
    """One structured JSON-able event per stage per turn. `payload` holds
    the event-specific fields (see the blueprint's M6 event table for the
    expected keys per `event` name, e.g. controller_decision needs
    latency_ms and utterance_progress_pct)."""

    trace_id: str
    session_id: str
    stage: str    # e.g. "controller", "decomposer", "retrieval", "synthesis"
    event: str    # e.g. "controller_decision", "retrieval_started"
    t_ms: float
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic v2 upgrade path (do this once you have pip/internet access)
# ---------------------------------------------------------------------------
# 1. pip install pydantic
# 2. Change `@dataclass` -> `class X(BaseModel):` and drop the decorator.
# 3. `field(default_factory=list)` -> `Field(default_factory=list)`.
# 4. `Decision`/`ClaimStatus` as `str, Enum` subclasses work unchanged as
#    Pydantic field types.
# 5. Biggest real gain: pass `ControllerDecision` (or any of these) as the
#    `response_model` / structured-output schema for your Stage 2 LLM call
#    in M1 — Pydantic validates the model's JSON output against this exact
#    shape and raises on drift, instead of you hand-parsing dict output.
# 6. Re-run the smoke test below (and demo_retrieval.py) after switching —
#    Pydantic is stricter about type coercion than a plain dataclass.


if __name__ == "__main__":
    # Smoke test: every schema instantiates and round-trips its key fields.
    # Run: python contracts.py
    tc = TranscriptChunk(session_id="s1", seq=0, t_ms=0.0, text_delta="I need to plan")
    assert tc.is_final is False

    cd = ControllerDecision(decision=Decision.WAIT, confidence=0.4, reason="incomplete")
    assert cd.decision == Decision.WAIT

    sq = SubQuery(id="q1", text="venue capacity in Pune", intent_type="constraint")
    assert sq.intent_type == "constraint"

    ec = EvidenceChunk(
        chunk_id="Doc_01::s1::c0", doc_id="Doc_01", section_ref="Section 1",
        heading_path="Doc_01 > Section 1", text="Venue A capacity is 30.",
    )
    assert ec.citation() == "[Doc_01 §Section 1]"

    claim = Claim(id="c1", text="Venue A seats 30.", citations=[ec.citation()])
    assert claim.status == ClaimStatus.ADDED

    av = AnswerVersion(version=1, claims=[claim], uncertainties=["catering unverifiable"])
    assert av.citations_all == ["[Doc_01 §Section 1]"]

    ev = TelemetryEvent(
        trace_id="t1", session_id="s1", stage="controller",
        event="controller_decision", t_ms=120.0, payload={"decision": "WAIT"},
    )
    assert ev.stage == "controller"

    print("All 7 contracts instantiate and validate correctly.")
