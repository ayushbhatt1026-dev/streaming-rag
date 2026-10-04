"""Run: python tests/test_refinement.py"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import AnswerVersion, ClaimStatus
from decomposer.backends import HeuristicDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from refinement.claim_builder import build_initial_claims
from refinement.refinement_engine import RefinementEngine
from retrieval.embeddings import TfidfEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import InMemoryVectorStore
from session.store import SessionStore
from telemetry.bus import TelemetryBus

CORPUS_DIR = str(Path(__file__).resolve().parents[1] / "corpus")


def _setup():
    eb = TfidfEmbeddingBackend()
    retriever = Retriever(embedding_backend=eb, vector_store=InMemoryVectorStore())
    retriever.index_corpus(corpus_dir=CORPUS_DIR)
    telemetry = TelemetryBus(jsonl_path="telemetry/test_refinement_log.jsonl")
    return eb, retriever, telemetry


def test_claims_persist_and_status_updates_correctly():
    eb, retriever, telemetry = _setup()
    session_id = "test-session-1"

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

    assert v1_ids <= set(v2_by_id.keys()), "all v1 claim ids must persist into v2"
    assert v2.version == v1.version + 1
    assert len(v2.claims) > len(v1.claims), "refine should add new claims for the delta"
    for cid in v1_ids:
        assert v2_by_id[cid].status == ClaimStatus.KEPT, (
            f"carried-forward claim {cid} should be KEPT in v2, "
            f"got {v2_by_id[cid].status}"
        )
        assert v2_by_id[cid].citations == [
            c.citations for c in v1.claims if c.id == cid
        ][0], "KEPT claim must preserve its ORIGINAL citations"


def test_refine_never_sees_the_original_utterance():
    """Proxy for G5's 'no full-corpus query during REFINE' assertion: the
    delta decomposition call must only ever receive the constraint text."""
    eb, retriever, telemetry = _setup()
    session_id = "test-session-2"
    original_utterance = "Summarize the travel reimbursement rule for an employee trip"

    trace1 = str(uuid.uuid4())
    decomposer1 = MultiIntentDecomposer(
        backend=HeuristicDecomposeBackend(), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace1,
    )
    sqs = decomposer1.decompose(original_utterance, trace1)
    evidence = retriever.search(sqs)
    v1 = build_initial_claims(sqs, evidence)

    seen_texts = []

    class _SpyBackend:
        def __init__(self, inner):
            self._inner = inner

        def propose(self, utterance):
            seen_texts.append(utterance)
            return self._inner.propose(utterance)

    trace2 = str(uuid.uuid4())
    decomposer2 = MultiIntentDecomposer(
        backend=_SpyBackend(HeuristicDecomposeBackend()), embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    engine = RefinementEngine(
        decomposer=decomposer2, retriever=retriever, embedding_backend=eb,
        telemetry=telemetry, session_id=session_id, trace_id=trace2,
    )
    constraint = "The trip was international and the booking was made after travel"
    engine.refine(v1, constraint)

    assert all(original_utterance not in t for t in seen_texts), (
        f"delta decompose leaked the original utterance: {seen_texts}"
    )


def test_reset_session_clears_state():
    store = SessionStore()
    store.set("s1", AnswerVersion(version=1, claims=[], uncertainties=[]))
    assert store.get("s1") is not None
    store.reset_session("s1")
    assert store.get("s1") is None


if __name__ == "__main__":
    test_claims_persist_and_status_updates_correctly()
    test_refine_never_sees_the_original_utterance()
    test_reset_session_clears_state()
    print("All refinement tests passed.")
