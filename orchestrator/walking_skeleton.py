"""
Phase 1, Step 4 — the walking skeleton, and Phase 2, Steps 5-6 — the real
Controller (M1) and Decomposer (M2) plugged into it, rather than throwaway
mocks. One simulated utterance goes in; a trace_id, a sequence of controller
decisions, and (if retrieval fired) a cited answer come out.

This intentionally skips M4 (session refinement) and the real M5 (grounded
LLM synthesis) — those are later steps. REFINE decisions are not produced
by this controller yet (see controller/controller.py's docstring) and
would currently just be ignored if they occurred.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from contracts import Decision, EvidenceChunk, TelemetryEvent
from controller.controller import RetrievalController
from decomposer.decomposer import MultiIntentDecomposer
from retrieval.retriever import Retriever
from simulator.transcript_simulator import simulate_utterance
from synthesis.naive_synthesizer import NaiveSynthesizer
from telemetry.bus import TelemetryBus


@dataclass
class TurnResult:
    trace_id: str
    session_id: str
    final_decision: Optional[Decision]
    sub_query_texts: List[str] = field(default_factory=list)
    evidence: List[EvidenceChunk] = field(default_factory=list)
    answer: str = ""
    n_retrievals_fired: int = 0
    early_retrieval: bool = False  # did retrieval start before the final chunk?


def run_one_utterance(
    utterance: str,
    controller: RetrievalController,
    decomposer: MultiIntentDecomposer,
    retriever: Retriever,
    synthesizer: NaiveSynthesizer,
    telemetry: TelemetryBus,
    session_id: str,
    trace_id: str,
) -> TurnResult:
    chunks = simulate_utterance(utterance, session_id=session_id)
    result = TurnResult(trace_id=trace_id, session_id=session_id, final_decision=None)

    last_retrieved_sub_queries: List[str] = []
    all_evidence: List[EvidenceChunk] = []

    for chunk in chunks:
        telemetry.emit(
            TelemetryEvent(
                trace_id=trace_id, session_id=session_id, stage="stream_buffer",
                event="chunk_received", t_ms=chunk.t_ms,
                payload={"seq": chunk.seq, "text_delta": chunk.text_delta, "is_final": chunk.is_final},
            )
        )

        decision = controller.decide(chunk)
        result.final_decision = decision.decision

        if decision.decision in (Decision.RETRIEVE_PROVISIONAL, Decision.RETRIEVE_FINAL):
            if not chunk.is_final and not result.early_retrieval:
                result.early_retrieval = True  # first retrieval fired before is_final

            telemetry.emit(
                TelemetryEvent(
                    trace_id=trace_id, session_id=session_id, stage="retrieval",
                    event="retrieval_started", t_ms=chunk.t_ms,
                    payload={
                        "trigger": "provisional" if decision.decision == Decision.RETRIEVE_PROVISIONAL else "final",
                        "utterance_so_far": controller.utterance_so_far,
                    },
                )
            )

            sub_queries = decomposer.decompose(
                controller.utterance_so_far, parent_utterance_id=trace_id
            )
            last_retrieved_sub_queries = [sq.text for sq in sub_queries]

            evidence = retriever.search(sub_queries)
            all_evidence = evidence
            result.n_retrievals_fired += 1

            telemetry.emit(
                TelemetryEvent(
                    trace_id=trace_id, session_id=session_id, stage="retrieval",
                    event="retrieval_completed", t_ms=chunk.t_ms,
                    payload={"n_candidates": len(evidence)},
                )
            )

            if decision.decision == Decision.RETRIEVE_FINAL:
                answer = synthesizer.synthesize(sub_queries, evidence)
                result.answer = answer
                result.sub_query_texts = last_retrieved_sub_queries
                result.evidence = all_evidence

                telemetry.emit(
                    TelemetryEvent(
                        trace_id=trace_id, session_id=session_id, stage="synthesis",
                        event="turn_completed", t_ms=chunk.t_ms,
                        payload={"answer_preview": answer[:200]},
                    )
                )
                return result

        elif decision.decision == Decision.SUPPRESS:
            result.answer = "(suppressed — no retrieval fired for this turn)"
            telemetry.emit(
                TelemetryEvent(
                    trace_id=trace_id, session_id=session_id, stage="synthesis",
                    event="turn_completed", t_ms=chunk.t_ms,
                    payload={"answer_preview": result.answer},
                )
            )
            return result

    # Reached end of chunks without a RETRIEVE_FINAL or SUPPRESS (e.g. the
    # controller stayed provisional the whole time). Synthesize from
    # whatever evidence we last had, so a turn never silently produces
    # nothing.
    if last_retrieved_sub_queries:
        result.answer = synthesizer.synthesize(
            [sq for sq in decomposer.decompose(controller.utterance_so_far, trace_id)],
            all_evidence,
        )
        result.sub_query_texts = last_retrieved_sub_queries
        result.evidence = all_evidence
    else:
        result.answer = "(utterance ended without a retrieval decision — check controller tuning)"

    return result
