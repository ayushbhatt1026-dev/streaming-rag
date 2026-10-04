"""
eval/ablations.py — Two required ablation studies for Step 9.

Ablation 1: Hybrid (dense+sparse+RRF) vs. Dense-only retrieval
  Measures recall@5 (fraction of queries where expected doc appears in top-5)
  and groundedness (fraction of answers with >0 supported claims).

Ablation 2: Rule-based (Heuristic) vs. LLM-based (Groq) controller
  Measures early-retrieval rate, false-trigger rate, avg latency per decision,
  and estimated cost per LLM call.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from controller.controller import RetrievalController
from controller.stage2 import GroqStage2Confirm, HeuristicStage2Confirm
from decomposer.backends import GroqDecomposeBackend, HeuristicDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from orchestrator.walking_skeleton import run_one_utterance, TurnResult
from retrieval.embeddings import EmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore
from synthesis.grounded_synthesizer import GroundedSynthesizer
from synthesis.naive_synthesizer import NaiveSynthesizer
from telemetry.bus import TelemetryBus
from tests.labelled_sets import EARLY_ELIGIBLE, PRESENTATION_ONLY, INCOMPLETE, heldout


# ---------------------------------------------------------------------------
# Oracle: query → expected doc_id prefix (for recall measurement)
# ---------------------------------------------------------------------------
# These are ground-truth mappings based on the corpus content.
_ORACLE: Dict[str, str] = {
    "venue capacity":            "Doc_01",
    "Venue B capacity":          "Doc_01",
    "Venue A capacity":          "Doc_01",
    "cancellation policy":       "Doc_02",
    "catering options":          "Doc_03",
    "travel reimbursement":      "Doc_04",
    "hotel reimbursement":       "Doc_04",
    "international travel":      "Doc_04",
    "parking":                   "Doc_01",
}

_ABLATION_QUERIES = [
    "What is the seated capacity of Venue B?",
    "What is the cancellation policy for workshops?",
    "How do I claim travel reimbursement?",
]

_QUERY_EXPECTED_DOCS = [
    "Doc_01",  # venue capacity
    "Doc_02",  # cancellation
    "Doc_04",  # reimbursement
]


def _run_retrieval_only(query: str, retriever: Retriever):
    """Run one query through the retriever and return top-k chunks."""
    from contracts import SubQuery
    sq = SubQuery(id="sq1", text=query)
    return retriever.search([sq])


def _recall_at_k(chunks, expected_doc_prefix: str, k: int = 5) -> bool:
    """True if any of the top-k chunks belongs to the expected doc."""
    top_k = sorted(chunks, key=lambda c: -(c.rerank_score or c.rrf_score))[:k]
    return any(c.doc_id.startswith(expected_doc_prefix) for c in top_k)


# ---------------------------------------------------------------------------
# Ablation 1: Hybrid vs Dense-only retrieval
# ---------------------------------------------------------------------------

def ablation_retrieval_hybrid_vs_dense(
    hybrid_retriever: Retriever,
    groq_client,
) -> Dict[str, Any]:
    """
    Compare hybrid (dense+BM25+RRF) vs. dense-only on 5 representative queries.

    Reports:
      - recall@5 for both modes
      - groundedness delta (both require GroundedSynthesizer)
    """
    # Build a dense-only retriever by setting top_k_sparse=0
    dense_retriever = Retriever(
        embedding_backend=hybrid_retriever._embed,
        vector_store=hybrid_retriever._vector_store,
        reranker=hybrid_retriever._reranker,
        top_k_dense=hybrid_retriever.top_k_dense,
        top_k_sparse=0,
        top_k_final=hybrid_retriever.top_k_final,
    )
    dense_retriever._chunks_by_id = hybrid_retriever._chunks_by_id
    dense_retriever._bm25 = hybrid_retriever._bm25

    synthesizer = GroundedSynthesizer(llm_client=groq_client)
    stage2 = GroqStage2Confirm(client=groq_client)
    decomp_backend = GroqDecomposeBackend(client=groq_client)

    hybrid_recall = []
    dense_recall = []
    hybrid_supported = []
    dense_supported = []

    for query, expected_doc in zip(_ABLATION_QUERIES, _QUERY_EXPECTED_DOCS):
        # Hybrid retrieval
        hybrid_chunks = _run_retrieval_only(query, hybrid_retriever)
        hybrid_recall.append(_recall_at_k(hybrid_chunks, expected_doc, k=5))

        # Dense-only retrieval
        dense_retriever._cache = {}
        dense_chunks = _run_retrieval_only(query, dense_retriever)
        dense_recall.append(_recall_at_k(dense_chunks, expected_doc, k=5))

        # Groundedness: run full synthesis, count supported claims
        from contracts import SubQuery
        sq = SubQuery(id="sq1", text=query)

        h_text, h_av = synthesizer.synthesize_with_claims([sq], hybrid_chunks)
        hybrid_supported.append(len([c for c in h_av.claims if c.citations]))

        d_text, d_av = synthesizer.synthesize_with_claims([sq], dense_chunks)
        dense_supported.append(len([c for c in d_av.claims if c.citations]))

    hybrid_recall_rate = sum(hybrid_recall) / len(hybrid_recall)
    dense_recall_rate = sum(dense_recall) / len(dense_recall)
    hybrid_ground_avg = sum(hybrid_supported) / len(hybrid_supported)
    dense_ground_avg = sum(dense_supported) / len(dense_supported)

    return {
        "ablation": "retrieval_hybrid_vs_dense",
        "queries": len(_ABLATION_QUERIES),
        "hybrid": {
            "recall_at_5": hybrid_recall_rate,
            "avg_supported_claims": hybrid_ground_avg,
            "per_query_recall": hybrid_recall,
        },
        "dense_only": {
            "recall_at_5": dense_recall_rate,
            "avg_supported_claims": dense_ground_avg,
            "per_query_recall": dense_recall,
        },
        "recall_delta": hybrid_recall_rate - dense_recall_rate,
        "groundedness_delta": hybrid_ground_avg - dense_ground_avg,
        "winner": "hybrid" if hybrid_recall_rate >= dense_recall_rate else "dense_only",
    }


# ---------------------------------------------------------------------------
# Ablation 2: Rule-based vs LLM-based controller
# ---------------------------------------------------------------------------

def ablation_controller_rule_vs_llm(
    retriever: Retriever,
    groq_client,
) -> Dict[str, Any]:
    """
    Compare HeuristicStage2Confirm (rule-based) vs. GroqStage2Confirm (LLM-based)
    on held-out EARLY_ELIGIBLE and held-out no-retrieval sets.

    Reports per backend:
      - early_retrieval_rate (on EARLY_ELIGIBLE held-out)
      - false_trigger_rate (on PRESENTATION_ONLY + INCOMPLETE held-out)
      - avg_decision_latency_ms
      - estimated_cost_per_call (Groq: $0.01/1K tokens ≈ 0.02 cents/call; Heuristic: $0)
    """
    synthesizer = NaiveSynthesizer()  # use naive synth to keep ablation fast
    emb = retriever._embed

    early_held = heldout(EARLY_ELIGIBLE)
    no_ret_held = heldout(PRESENTATION_ONLY) + heldout(INCOMPLETE)

    results = {}

    for label, stage2_cls, cost_per_call in [
        ("heuristic", HeuristicStage2Confirm, 0.0),
        ("groq_llm",  GroqStage2Confirm,      0.0002),  # ~$0.0002 per call at $0.10/1K tok
    ]:
        stage2 = stage2_cls() if label == "heuristic" else stage2_cls(client=groq_client)
        decomp_backend = HeuristicDecomposeBackend()

        early_results = []
        early_latencies = []

        for utt in early_held:
            tel = TelemetryBus()
            sid = f"abl-{label}-{uuid.uuid4().hex[:6]}"
            tid = f"t-{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            ctrl = RetrievalController(stage2=stage2, telemetry=tel, session_id=sid, trace_id=tid)
            decomp = MultiIntentDecomposer(backend=decomp_backend, embedding_backend=emb,
                                           telemetry=tel, session_id=sid, trace_id=tid)
            res = run_one_utterance(
                utterance=utt, controller=ctrl, decomposer=decomp,
                retriever=retriever, synthesizer=synthesizer,
                telemetry=tel, session_id=sid, trace_id=tid,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            early_results.append(res)
            early_latencies.append(elapsed_ms)

        false_trigger = []
        pres_held = heldout(PRESENTATION_ONLY)
        incomp_held = heldout(INCOMPLETE)

        for utt in pres_held:
            tel = TelemetryBus()
            sid = f"abl-{label}-{uuid.uuid4().hex[:6]}"
            tid = f"t-{uuid.uuid4().hex[:8]}"
            ctrl = RetrievalController(stage2=stage2, telemetry=tel, session_id=sid, trace_id=tid)
            decomp = MultiIntentDecomposer(backend=decomp_backend, embedding_backend=emb,
                                           telemetry=tel, session_id=sid, trace_id=tid)
            res = run_one_utterance(
                utterance=utt, controller=ctrl, decomposer=decomp,
                retriever=retriever, synthesizer=synthesizer,
                telemetry=tel, session_id=sid, trace_id=tid,
                final_on_last=True,
            )
            false_trigger.append(res.n_retrievals_fired > 0)

        for utt in incomp_held:
            tel = TelemetryBus()
            sid = f"abl-{label}-{uuid.uuid4().hex[:6]}"
            tid = f"t-{uuid.uuid4().hex[:8]}"
            ctrl = RetrievalController(stage2=stage2, telemetry=tel, session_id=sid, trace_id=tid)
            decomp = MultiIntentDecomposer(backend=decomp_backend, embedding_backend=emb,
                                           telemetry=tel, session_id=sid, trace_id=tid)
            res = run_one_utterance(
                utterance=utt, controller=ctrl, decomposer=decomp,
                retriever=retriever, synthesizer=synthesizer,
                telemetry=tel, session_id=sid, trace_id=tid,
                final_on_last=False,
            )
            false_trigger.append(res.n_retrievals_fired > 0)

        n_early = len(early_results)
        n_noret = len(false_trigger)
        results[label] = {
            "early_retrieval_rate": sum(r.early_retrieval for r in early_results) / n_early if n_early else 0,
            "false_trigger_rate": sum(false_trigger) / n_noret if n_noret else 0,
            "avg_latency_ms": sum(early_latencies) / len(early_latencies) if early_latencies else 0,
            "cost_per_call_usd": cost_per_call,
            "n_early_tested": n_early,
            "n_noret_tested": n_noret,
        }

    return {
        "ablation": "controller_rule_vs_llm",
        "heuristic": results["heuristic"],
        "groq_llm": results["groq_llm"],
        "early_retrieval_delta": results["groq_llm"]["early_retrieval_rate"] - results["heuristic"]["early_retrieval_rate"],
        "false_trigger_delta": results["groq_llm"]["false_trigger_rate"] - results["heuristic"]["false_trigger_rate"],
        "latency_delta_ms": results["groq_llm"]["avg_latency_ms"] - results["heuristic"]["avg_latency_ms"],
    }
