"""
Step 7 — Claim-Graph Refinement Engine & Session Store (Gate G5 & G6).

Requirements:
  R1  v1 Claim graph: initial claims built with ClaimStatus.ADDED, each tagged with citations
  R2  Refinement (M4): carries forward untouched claims as ClaimStatus.KEPT preserving original citations
  R3  Refine delta: new facts from constraint added as ClaimStatus.ADDED with incremented version
  R4  G5 guarantee: delta retrieval never receives the original utterance (structural isolation)
  R5  G5 query log: Retriever.query_log proves no full-corpus re-query was executed during REFINE
  R6  SessionStore: in-memory only, TTL-expiring, with explicit reset_session() teardown
  R7  Orchestrator REFINE branch: run_one_utterance executes delta refinement, produces AnswerVersion,
      updates SessionStore, and emits turn_completed telemetry (fixing G5/G6 fall-through)
  R8  Precedence fix: SUPPRESS does not beat refinement when correction cues are present
  R9  Cue strictness: loose discourse markers ("also", "however") do not falsely trigger REFINE
  R10 Config purity: hold_provisional_when_prior_answer is governed by config/controller.yaml

Run: pytest tests/test_step7_refinement.py -v
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
import pytest
import yaml

from contracts import AnswerVersion, Claim, ClaimStatus, Decision, SubQuery
from controller.controller import RetrievalController
from controller.rule_gate import RuleGate
from controller.stage2 import HeuristicStage2Confirm
from decomposer.backends import HeuristicDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from orchestrator.walking_skeleton import TurnResult, run_one_utterance
from refinement.claim_builder import build_initial_claims
from refinement.refinement_engine import RefinementEngine
from refinement.session_store import SessionStore
from retrieval.embeddings import TfidfEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import InMemoryVectorStore
from synthesis.naive_synthesizer import NaiveSynthesizer
from telemetry.bus import TelemetryBus

CORPUS_DIR = str(Path(__file__).resolve().parents[1] / "corpus")


def _setup_retriever():
    eb = TfidfEmbeddingBackend()
    retriever = Retriever(embedding_backend=eb, vector_store=InMemoryVectorStore())
    retriever.index_corpus(corpus_dir=CORPUS_DIR)
    return eb, retriever


# ------------------------------------------------------------- R1 & R2 & R3
def test_claims_persist_and_status_updates_correctly():
    eb, retriever = _setup_retriever()
    telemetry = TelemetryBus(jsonl_path="telemetry/test_step7_telemetry.jsonl")
    session_id = "s-persist-1"

    trace1 = str(uuid.uuid4())
    decomposer1 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace1,
    )
    utterance = "Summarize the travel reimbursement rule for an employee trip"
    sqs = decomposer1.decompose(utterance, trace1)
    evidence = retriever.search(sqs)
    v1 = build_initial_claims(sqs, evidence)
    assert len(v1.claims) >= 1
    assert all(c.status == ClaimStatus.ADDED for c in v1.claims)

    trace2 = str(uuid.uuid4())
    decomposer2 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    engine = RefinementEngine(
        decomposer=decomposer2, retriever=retriever, embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    constraint = "The trip was international and the booking was made after travel"
    v2 = engine.refine(v1, constraint)

    v1_ids = {c.id for c in v1.claims}
    v2_by_id = {c.id: c for c in v2.claims}

    assert v1_ids <= set(v2_by_id.keys()), "all v1 claim IDs must persist into v2"
    assert v2.version == v1.version + 1
    assert len(v2.claims) > len(v1.claims), "refine should add new claims for the delta"
    for cid in v1_ids:
        assert v2_by_id[cid].status == ClaimStatus.KEPT, (
            f"claim {cid} should be KEPT in v2, got {v2_by_id[cid].status}"
        )
        assert v2_by_id[cid].citations == [
            c.citations for c in v1.claims if c.id == cid
        ][0], "KEPT claim must preserve original citations"


# ------------------------------------------------------------- R4 & R5
def test_g5_retriever_query_log_proves_delta_only_search():
    eb, retriever = _setup_retriever()
    telemetry = TelemetryBus(jsonl_path="telemetry/test_step7_telemetry.jsonl")
    session_id = "s-g5-querylog"
    original_utterance = "Summarize the travel reimbursement rule for an employee trip"

    trace1 = str(uuid.uuid4())
    decomposer1 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace1,
    )
    sqs1 = decomposer1.decompose(original_utterance, trace1)
    evidence1 = retriever.search(sqs1)
    v1 = build_initial_claims(sqs1, evidence1)

    # Clear query log before refinement turn
    retriever.clear_query_log()
    assert len(retriever.query_log) == 0

    trace2 = str(uuid.uuid4())
    decomposer2 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    engine = RefinementEngine(
        decomposer=decomposer2, retriever=retriever, embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    constraint = "actually international flight booking requires pre-approval"
    v2 = engine.refine(v1, constraint)

    # Assert G5 via query log:
    # 1. Queries were logged
    assert len(retriever.query_log) > 0
    # 2. None of the logged queries contain the original turn's prompt
    for q_entry in retriever.query_log:
        q_text = q_entry["sub_query"].lower()
        assert "summarize the travel reimbursement rule for an employee trip" not in q_text
        assert "sub_query_id" in q_entry
        assert "cache_hit" in q_entry


# ------------------------------------------------------------- R6
def test_session_store_ttl_and_reset():
    # Test TTL expiration
    store = SessionStore(ttl_seconds=0.05)
    v1 = AnswerVersion(version=1, claims=[], uncertainties=[])
    store.set("sess-ttl", v1)
    assert store.get("sess-ttl") is not None
    time.sleep(0.06)
    assert store.get("sess-ttl") is None

    # Test explicit reset_session()
    store2 = SessionStore(ttl_seconds=3600.0)
    store2.set("sess-reset", v1)
    assert store2.get("sess-reset") is not None
    store2.reset_session("sess-reset")
    assert store2.get("sess-reset") is None


# ------------------------------------------------------------- R7
def test_orchestrator_handles_refine_and_emits_turn_completed():
    eb, retriever = _setup_retriever()
    telemetry = TelemetryBus(jsonl_path="telemetry/test_step7_telemetry.jsonl")
    session_store = SessionStore()
    synthesizer = NaiveSynthesizer()
    session_id = f"sess-orch-{uuid.uuid4().hex[:6]}"

    # --- Turn 1: normal retrieval ---
    trace_id_1 = str(uuid.uuid4())
    controller1 = RetrievalController(
        stage2=HeuristicStage2Confirm(), telemetry=telemetry,
        session_id=session_id, trace_id=trace_id_1,
    )
    decomposer1 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace_id_1,
    )
    t1_res = run_one_utterance(
        utterance="What is the cancellation policy for workshops in Pune?",
        controller=controller1,
        decomposer=decomposer1,
        retriever=retriever,
        synthesizer=synthesizer,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id_1,
        session_store=session_store,
    )

    assert t1_res.final_decision == Decision.RETRIEVE_FINAL
    assert t1_res.answer_version is not None
    assert t1_res.answer_version.version == 1
    assert session_store.get(session_id) is not None

    # Verify Turn 1 emitted turn_completed
    t1_events = telemetry.events_for_trace(trace_id_1)
    assert any(e["event"] == "turn_completed" for e in t1_events)

    # --- Turn 2: refinement turn ---
    trace_id_2 = str(uuid.uuid4())
    controller2 = RetrievalController(
        stage2=HeuristicStage2Confirm(), telemetry=telemetry,
        session_id=session_id, trace_id=trace_id_2,
        prior_context="cancellation policy for workshops in Pune",
    )
    decomposer2 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace_id_2,
    )
    refinement_engine = RefinementEngine(
        decomposer=decomposer2, retriever=retriever, embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace_id_2,
    )

    t2_res = run_one_utterance(
        utterance="Actually Venue B instead for 50 people",
        controller=controller2,
        decomposer=decomposer2,
        retriever=retriever,
        synthesizer=synthesizer,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id_2,
        session_store=session_store,
        refinement_engine=refinement_engine,
    )

    # Must NOT fall through to "(utterance ended without a retrieval decision)"
    assert t2_res.final_decision == Decision.REFINE
    assert "utterance ended without a retrieval decision" not in t2_res.answer
    assert t2_res.answer_version is not None
    assert t2_res.answer_version.version == 2

    # Verify Turn 2 emitted turn_completed
    t2_events = telemetry.events_for_trace(trace_id_2)
    assert any(e["event"] == "turn_completed" for e in t2_events), "turn_completed must be emitted on REFINE"


# ------------------------------------------------------------- R8
def test_suppress_does_not_beat_refinement_with_prior_context():
    # If an utterance contains both presentation words ("summarize") AND a refinement cue ("actually"),
    # RuleGate should NOT return SUPPRESS.
    gate = RuleGate()
    decision = gate.evaluate("Actually summarize that for Mumbai instead")
    assert decision != Decision.SUPPRESS, "Refinement cue must prevent premature SUPPRESS"


# ------------------------------------------------------------- R9
def test_heuristic_refinement_cue_strictness():
    stage2 = HeuristicStage2Confirm()
    # Unrelated queries with loose connectors should NOT be marked as refinement
    d_also = stage2.confirm("Also what is the parking policy", is_final=True, prior_context="travel rules")
    assert d_also.is_refinement_of_prior_answer is False

    d_however = stage2.confirm("However what about the hotel allowance", is_final=True, prior_context="travel rules")
    assert d_however.is_refinement_of_prior_answer is False

    # Genuine correction cues DO mark as refinement
    d_act = stage2.confirm("Actually the trip was domestic", is_final=True, prior_context="travel rules")
    assert d_act.is_refinement_of_prior_answer is True

    d_instead = stage2.confirm("Book Venue B instead", is_final=True, prior_context="venue capacity")
    assert d_instead.is_refinement_of_prior_answer is True


# ------------------------------------------------------------- R10
def test_hold_provisional_when_prior_answer_in_config():
    with open("config/controller.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "hold_provisional_when_prior_answer" in cfg
    assert isinstance(cfg["hold_provisional_when_prior_answer"], bool)
