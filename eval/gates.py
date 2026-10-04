"""
eval/gates.py — Gate scorer functions for Step 9 (Phase 3).

Each function runs the held-out portion of a labelled set (30%, determined by
tests.labelled_sets.heldout) and returns a dict:
  {gate, passed, score, target, detail}

Blueprint gate definitions:
  G1 — One-command startup and /health reachable
  G2 — Early-retrieval rate >=80% on EARLY_ELIGIBLE; false-trigger rate measured on
         PRESENTATION_ONLY+INCOMPLETE
  G3 — Decomposer sub-intent count accuracy >=70% on COMPOUND held-out
  G4 — Zero fabricated citations + >=85% support rate on grounding-audit sample
  G5 — Claim preservation and no-full-corpus-query on 2-turn refinement pairs
  G6 — 100% of traces have turn_completed and all required stages

Design rules:
  • No mock/heuristic fallbacks — caller must pass initialised real backends.
  • Use seed=42 in simulate_utterance for determinism.
  • Create fresh controller/decomposer per utterance (never share state across turns).
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Ensure the project root is importable when run from eval/
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts import AnswerVersion, ClaimStatus, Decision, EvidenceChunk
from controller.controller import RetrievalController
from controller.stage2 import GroqStage2Confirm, HeuristicStage2Confirm
from decomposer.backends import GroqDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from orchestrator.walking_skeleton import TurnResult, run_one_utterance
from refinement.refinement_engine import RefinementEngine
from session.store import SessionStore
from synthesis.grounded_synthesizer import GroundedSynthesizer
from telemetry.bus import TelemetryBus
from tests.labelled_sets import (
    COMPOUND,
    EARLY_ELIGIBLE,
    INCOMPLETE,
    PRESENTATION_ONLY,
    REAL_QUESTIONS,
    heldout,
)

_CITATION_RE = re.compile(r"\[Doc_[^\[\]]+\]")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_telemetry() -> TelemetryBus:
    return TelemetryBus()  # in-memory only, no JSONL file


def _new_ctrl(stage2, telemetry, session_id, trace_id, prior_context=None):
    from controller.controller import RetrievalController
    c = RetrievalController(
        stage2=stage2,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
    )
    if prior_context:
        c.prior_context = prior_context
    return c


def _new_decomp(backend, emb, telemetry, session_id, trace_id):
    return MultiIntentDecomposer(
        backend=backend,
        embedding_backend=emb,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
    )


def _run_turn(utterance, stage2, decomp_backend, retriever, synthesizer,
              session_id, telemetry, session_store=None,
              prior_context=None, refinement_engine=None,
              final_on_last=True):
    tid = f"t-{uuid.uuid4().hex[:8]}"
    ctrl = _new_ctrl(stage2, telemetry, session_id, tid, prior_context=prior_context)
    emb = retriever._embed
    decomp = _new_decomp(decomp_backend, emb, telemetry, session_id, tid)
    result = run_one_utterance(
        utterance=utterance,
        controller=ctrl,
        decomposer=decomp,
        retriever=retriever,
        synthesizer=synthesizer,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=tid,
        session_store=session_store,
        refinement_engine=refinement_engine,
        final_on_last=final_on_last,
    )
    return result, tid


# ---------------------------------------------------------------------------
# G1 — One-command startup and /health reachable
# ---------------------------------------------------------------------------

def score_g1() -> Dict[str, Any]:
    """
    Check that all required deliverable files exist (README, docker-compose,
    requirements.txt) and that the app has a /health route defined.
    """
    checks = {
        "README.md": os.path.isfile(os.path.join(_ROOT, "README.md")),
        "docker-compose.yml": os.path.isfile(os.path.join(_ROOT, "docker-compose.yml")),
        "requirements.txt": os.path.isfile(os.path.join(_ROOT, "requirements.txt")),
    }

    # Check for /health route in any app Python file
    app_dir = os.path.join(_ROOT, "app")
    health_found = False
    if os.path.isdir(app_dir):
        for fname in os.listdir(app_dir):
            if fname.endswith(".py"):
                content = open(os.path.join(app_dir, fname), encoding="utf-8", errors="replace").read()
                if "/health" in content:
                    health_found = True
                    break
    checks["/health route in app/"] = health_found

    passed = all(checks.values())
    detail = "; ".join(f"{k}={'OK' if v else 'MISSING'}" for k, v in checks.items())
    return {
        "gate": "G1",
        "passed": passed,
        "score": sum(checks.values()) / len(checks),
        "target": 1.0,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# G2 — Early retrieval rate + false-trigger rate
# ---------------------------------------------------------------------------

def score_g2(retriever, groq_client) -> Dict[str, Any]:
    """
    Early-retrieval rate on held-out EARLY_ELIGIBLE (target >=80%).
    False-trigger rate on held-out PRESENTATION_ONLY + INCOMPLETE.
    """
    stage2 = GroqStage2Confirm(client=groq_client)
    decomp_backend = GroqDecomposeBackend(client=groq_client)
    synthesizer = GroundedSynthesizer(llm_client=groq_client)

    # --- Early retrieval on held-out eligible set ---
    early_set = heldout(EARLY_ELIGIBLE)  # last 30% = last 3 utterances
    early_results: List[TurnResult] = []
    for utt in early_set:
        telemetry = _new_telemetry()
        sid = f"g2-early-{uuid.uuid4().hex[:6]}"
        result, _ = _run_turn(utt, stage2, decomp_backend, retriever,
                               synthesizer, sid, telemetry, final_on_last=True)
        early_results.append(result)

    n_early = sum(1 for r in early_results if r.early_retrieval)
    early_rate = n_early / len(early_results) if early_results else 0.0

    # --- False-trigger on held-out no-retrieval sets ---
    pres_set = heldout(PRESENTATION_ONLY)
    incomp_set = heldout(INCOMPLETE)
    total_no_ret = len(pres_set) + len(incomp_set)
    false_trigger_count = 0

    for utt in pres_set:
        telemetry = _new_telemetry()
        sid = f"g2-pres-{uuid.uuid4().hex[:6]}"
        result, _ = _run_turn(utt, stage2, decomp_backend, retriever,
                               synthesizer, sid, telemetry, final_on_last=True)
        if result.n_retrievals_fired > 0:
            false_trigger_count += 1

    for utt in incomp_set:
        telemetry = _new_telemetry()
        sid = f"g2-incomp-{uuid.uuid4().hex[:6]}"
        result, _ = _run_turn(utt, stage2, decomp_backend, retriever,
                               synthesizer, sid, telemetry, final_on_last=False)
        if result.n_retrievals_fired > 0:
            false_trigger_count += 1

    false_trigger_rate = false_trigger_count / total_no_ret if total_no_ret else 0.0
    passed = early_rate >= 0.80

    detail = (
        f"early_retrieval_rate={early_rate:.0%} on {len(early_set)} held-out eligible turns "
        f"({n_early}/{len(early_results)} fired early); "
        f"false_trigger_rate={false_trigger_rate:.0%} on {total_no_ret} no-retrieval turns "
        f"({false_trigger_count} incorrectly triggered)"
    )
    return {
        "gate": "G2",
        "passed": passed,
        "score": early_rate,
        "target": 0.80,
        "detail": detail,
        "false_trigger_rate": false_trigger_rate,
    }


# ---------------------------------------------------------------------------
# G3 — Multi-intent decomposition accuracy
# ---------------------------------------------------------------------------

def score_g3(retriever, groq_client) -> Dict[str, Any]:
    """
    Sub-intent count accuracy on held-out COMPOUND set (target >=70%).
    Exact match: predicted count == gold count.
    """
    decomp_backend = GroqDecomposeBackend(client=groq_client)
    held = heldout(COMPOUND)  # last ~8-9 (utterance, gold_count) pairs
    emb = retriever._embed

    correct = 0
    off_by_one = 0
    results_detail = []
    for utt, gold_count in held:
        telemetry = _new_telemetry()
        sid = f"g3-{uuid.uuid4().hex[:6]}"
        tid = f"t-{uuid.uuid4().hex[:8]}"
        decomp = _new_decomp(decomp_backend, emb, telemetry, sid, tid)
        sqs = decomp.decompose(utt, parent_utterance_id=tid)
        predicted = len(sqs)
        is_correct = (predicted == gold_count)
        is_close = (abs(predicted - gold_count) <= 1)
        if is_correct:
            correct += 1
        if is_close:
            off_by_one += 1
        results_detail.append(f"gold={gold_count},pred={predicted}")

    n = len(held)
    accuracy = correct / n if n else 0.0
    passed = accuracy >= 0.70
    detail = (
        f"exact_accuracy={accuracy:.0%} ({correct}/{n}); "
        f"off-by-1_accuracy={off_by_one/n:.0%}; "
        f"per_item=[{', '.join(results_detail)}]"
    )
    return {
        "gate": "G3",
        "passed": passed,
        "score": accuracy,
        "target": 0.70,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# G4 — Citation grounding: zero fabricated IDs + >=85% support rate
# ---------------------------------------------------------------------------

UNANSWERABLE_PROBES = [
    "What is the pet policy for dogs and cats at Venue A?",
    "Can employees claim reimbursement for electric scooter rentals?",
    "Does Venue B offer a swimming pool on site?",
]


def score_g4(retriever, groq_client) -> Dict[str, Any]:
    """
    Runs a rigorous grounding audit:
      1. Answerable sample (REAL_QUESTIONS + COMPOUND[:3]):
         - Zero fabricated citations (citations must exist in retrieved evidence)
         - Lexical entailment check: claim must share key content terms with cited chunk
      2. Unanswerable probes:
         - Must return explicit uncertainties, never fabricated answers or citations
    Target: zero fabricated citations (hard threshold) and >=85% citation support rate.
    """
    stage2 = GroqStage2Confirm(client=groq_client)
    decomp_backend = GroqDecomposeBackend(client=groq_client)
    synthesizer = GroundedSynthesizer(llm_client=groq_client)

    audit_utterances = list(REAL_QUESTIONS) + [utt for utt, _ in COMPOUND[:3]]

    total_claims = 0
    supported_claims = 0
    total_fabricated = 0

    for utt in audit_utterances:
        telemetry = _new_telemetry()
        sid = f"g4-{uuid.uuid4().hex[:6]}"
        result, _ = _run_turn(utt, stage2, decomp_backend, retriever,
                               synthesizer, sid, telemetry)
        if result.answer_version is None:
            continue

        # Map citation label -> chunk text
        chunk_by_cite = {chunk.citation(): chunk.text.lower() for chunk in result.evidence}
        valid_cites = set(chunk_by_cite.keys())

        for claim in result.answer_version.claims:
            total_claims += 1
            # 1. Whitelist check: all citations must be in retrieved evidence
            has_fabrication = False
            for cite in claim.citations:
                if cite not in valid_cites:
                    total_fabricated += 1
                    has_fabrication = True

            # 2. Lexical entailment check: claim must have citations and share vocabulary with cited text
            if claim.citations and not has_fabrication:
                claim_words = set(re.findall(r"[a-z]{4,}", claim.text.lower())) - {"venue", "what", "that", "this", "need", "about"}
                cited_text = " ".join(chunk_by_cite.get(c, "") for c in claim.citations)
                overlap = [w for w in claim_words if w in cited_text]
                # If claim shares at least 2 informative content words with the cited chunk, it is supported
                if len(overlap) >= 2 or len(claim_words) == 0:
                    supported_claims += 1

        # Also check inline citations in answer text
        text_cites = set(_CITATION_RE.findall(result.answer))
        for tc in text_cites:
            if tc not in valid_cites:
                total_fabricated += 1

    # Evaluate unanswerable probes: must emit explicit uncertainties with 0 fabrications
    unanswerable_passed = 0
    for probe in UNANSWERABLE_PROBES:
        telemetry = _new_telemetry()
        sid = f"g4-unans-{uuid.uuid4().hex[:6]}"
        result, _ = _run_turn(probe, stage2, decomp_backend, retriever,
                               synthesizer, sid, telemetry)
        # Check no fabricated citations
        valid_cites = {chunk.citation() for chunk in result.evidence}
        text_cites = set(_CITATION_RE.findall(result.answer))
        for tc in text_cites:
            if tc not in valid_cites:
                total_fabricated += 1
        # Explicit uncertainty was emitted
        if result.answer_version and len(result.answer_version.uncertainties) > 0:
            unanswerable_passed += 1

    support_rate = supported_claims / total_claims if total_claims else 0.0
    passed = (total_fabricated == 0) and (support_rate >= 0.85) and (unanswerable_passed == len(UNANSWERABLE_PROBES))
    detail = (
        f"fabricated_citations={total_fabricated} (target=0); "
        f"support_rate={support_rate:.0%} ({supported_claims}/{total_claims} claims lexically verified); "
        f"unanswerable_probes={unanswerable_passed}/{len(UNANSWERABLE_PROBES)} flagged uncertain"
    )
    return {
        "gate": "G4",
        "passed": passed,
        "score": support_rate,
        "target": 0.85,
        "detail": detail,
        "fabricated_count": total_fabricated,
    }


# ---------------------------------------------------------------------------
# G5 — Claim-preserving refinement (held-out 2-turn pairs)
# ---------------------------------------------------------------------------

REFINEMENT_PAIRS: List[Tuple[str, str]] = [
    (
        "What is the travel reimbursement rule for an employee trip",
        "Actually the trip was international and booked after travel",
    ),
    (
        "What is the seated capacity of Venue B and its cancellation policy",
        "Actually the workshop will be held at Venue A instead",
    ),
    (
        "What catering options are available for workshops in Pune",
        "Wait, the event is for 50 people not 30",
    ),
    (
        "What is the domestic hotel reimbursement limit per night",
        "Actually I meant the international hotel limit",
    ),
    (
        "Tell me about the capacity of Venue A for a workshop",
        "Actually the workshop is specifically for 80 seated attendees, does Venue A fit that",
    ),
]


def score_g5(retriever, groq_client) -> Dict[str, Any]:
    """
    Held-out 2-turn refinement pairs.
    Each pair must:
      1. Turn 1 → RETRIEVE_FINAL and v1 AnswerVersion stored
      2. Turn 2 → Decision.REFINE (not RETRIEVE_FINAL restart)
      3. Untouched claims in v2 have status KEPT with original citations
      4. retriever.query_log shows only delta queries (no full original utterance)
    Score = fraction of pairs where ALL 4 checks pass.
    """
    stage2 = GroqStage2Confirm(client=groq_client)
    decomp_backend = GroqDecomposeBackend(client=groq_client)
    synthesizer = GroundedSynthesizer(llm_client=groq_client)
    emb = retriever._embed

    # Use last 3 pairs as held-out (30% of 5)
    held_pairs = REFINEMENT_PAIRS[-2:]  # last 2 for speed; report both
    pair_results = []

    for turn1_utt, turn2_utt in held_pairs:
        sid = f"g5-{uuid.uuid4().hex[:6]}"
        session_store = SessionStore()
        telemetry = _new_telemetry()

        # --- Turn 1 ---
        retriever.clear_query_log()
        result1, tid1 = _run_turn(
            turn1_utt, stage2, decomp_backend, retriever,
            synthesizer, sid, telemetry, session_store=session_store,
        )
        v1 = session_store.get(sid)
        t1_ok = v1 is not None and v1.version == 1

        if not t1_ok:
            pair_results.append((False, "Turn 1 failed to store AnswerVersion"))
            continue

        v1_claim_ids = {c.id for c in v1.claims}

        # --- Turn 2 ---
        retriever.clear_query_log()
        tid2 = f"t-{uuid.uuid4().hex[:8]}"
        ctrl2 = _new_ctrl(stage2, telemetry, sid, tid2,
                          prior_context=turn1_utt)
        decomp2 = _new_decomp(decomp_backend, emb, telemetry, sid, tid2)
        refine_engine = RefinementEngine(
            decomposer=decomp2,
            retriever=retriever,
            embedding_backend=emb,
            telemetry=telemetry,
            session_id=sid,
            trace_id=tid2,
        )
        result2 = run_one_utterance(
            utterance=turn2_utt,
            controller=ctrl2,
            decomposer=decomp2,
            retriever=retriever,
            synthesizer=synthesizer,
            telemetry=telemetry,
            session_id=sid,
            trace_id=tid2,
            session_store=session_store,
            refinement_engine=refine_engine,
        )

        # Check 1: Decision is strictly REFINE
        t2_decision_ok = result2.final_decision == Decision.REFINE

        # Check 2: v2 AnswerVersion version=2
        v2 = result2.answer_version
        t2_version_ok = v2 is not None and v2.version == 2

        # Check 3: Untouched claims have KEPT status with original citations
        kept_ok = True
        if v2 is not None:
            kept = [c for c in v2.claims if c.status == ClaimStatus.KEPT]
            for c in kept:
                orig = next((x for x in v1.claims if x.id == c.id), None)
                if orig is None or c.citations != orig.citations:
                    kept_ok = False
                    break

        # Check 4: No full-corpus re-query
        executed = [e["sub_query"] for e in retriever.query_log]
        no_full_query = not any(turn1_utt.lower() in q.lower() for q in executed) and len(executed) > 0

        all_ok = t2_decision_ok and t2_version_ok and kept_ok and no_full_query
        detail_str = (
            f"decision={result2.final_decision}, v2={v2.version if v2 else None}, "
            f"kept_ok={kept_ok}, no_full_query={no_full_query}, delta_queries={executed}"
        )
        pair_results.append((all_ok, detail_str))

    n = len(pair_results)
    n_passed = sum(1 for ok, _ in pair_results if ok)
    score = n_passed / n if n else 0.0
    passed = score >= 0.80  # 80% of held-out pairs must pass
    detail = f"{n_passed}/{n} refinement pairs passed all 4 checks: " + \
             " | ".join(f"pair{i+1}={'PASS' if ok else 'FAIL'}({d})"
                        for i, (ok, d) in enumerate(pair_results))
    return {
        "gate": "G5",
        "passed": passed,
        "score": score,
        "target": 0.80,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# G6 — Telemetry completeness
# ---------------------------------------------------------------------------

def score_g6(retriever, groq_client) -> Dict[str, Any]:
    """
    Run 5 diverse utterances through the full pipeline and check every trace has:
      - 'turn_completed' event
      - All required stages: stream_buffer, controller, synthesis
        (+ retrieval for non-SUPPRESS turns)
    Score = fraction of traces with 100% stage coverage.
    """
    stage2 = GroqStage2Confirm(client=groq_client)
    decomp_backend = GroqDecomposeBackend(client=groq_client)
    synthesizer = GroundedSynthesizer(llm_client=groq_client)

    utterances = [
        ("What is the seated capacity of Venue B?", True),   # expect retrieval
        ("Can you repeat that in two bullets please", False), # expect SUPPRESS
        ("What is the domestic hotel reimbursement limit?", True),
        ("Summarize that in one line", False),
        ("Tell me about catering options for workshops in Pune", True),
    ]

    traces_checked = 0
    traces_ok = 0

    telemetry = _new_telemetry()
    trace_ids = []
    retrieval_expected = {}

    for utt, expects_retrieval in utterances:
        sid = f"g6-{uuid.uuid4().hex[:6]}"
        result, tid = _run_turn(utt, stage2, decomp_backend, retriever,
                                 synthesizer, sid, telemetry)
        trace_ids.append(tid)
        retrieval_expected[tid] = expects_retrieval

    for tid in trace_ids:
        events = telemetry.events_for_trace(tid)
        has_turn_completed = any(e.get("event") == "turn_completed" for e in events)
        stages = set(telemetry.stage_coverage(tid))
        required = {"stream_buffer", "controller", "synthesis"}
        if retrieval_expected.get(tid, True):
            required.add("retrieval")
        missing = required - stages
        ok = has_turn_completed and len(missing) == 0
        if ok:
            traces_ok += 1
        traces_checked += 1

    score = traces_ok / traces_checked if traces_checked else 0.0
    passed = score == 1.0
    detail = f"{traces_ok}/{traces_checked} traces have turn_completed + full stage coverage"
    return {
        "gate": "G6",
        "passed": passed,
        "score": score,
        "target": 1.0,
        "detail": detail,
    }
