"""
Step 6 — multi-intent decomposition + anti-fragmentation guard (Gate G3). Requirements:

  R1  single intent -> exactly ONE sub-query (never padded)
  R2  compound utterance -> 2-4 distinct, non-overlapping sub-queries
  R3  near-duplicate candidates merge (cosine >= threshold)
  R4  hard cap of 4 sub-queries, keeping the most distinct
  R5  every SubQuery is well-formed (unique id, text, valid intent_type, parent id)
  R6  threshold + cap come from config/decomposer.yaml
  R7  decomposition_completed telemetry (n_raw / n_after_merge / n_final / latency)
  R8  backend failures fail SAFE (bad JSON -> whole utterance; outage -> heuristic)
  R9  each sub-query actually retrieves evidence (hand-off to M3 works)
  R10 G3 metric on a held-out labelled set (>=70% with the real LLM backend)

Run: python -m pytest tests/test_step6_decomposer.py -v -s
Optional real-LLM checks:  $env:GEMINI_API_KEY="..." ; $env:DECOMPOSER_BACKEND="gemini"
"""

import os
import re

import numpy as np
import pytest

from conftest import FakeLLMClient
from decomposer.backends import GeminiDecomposeBackend, HeuristicDecomposeBackend
from labelled_sets import COMPOUND, heldout

VALID_INTENTS = {"factual_lookup", "policy", "constraint", "comparison"}
DEMO_COMPOUND = "I need to plan a customer workshop in Pune for 30 people and I need the cancellation policy and the catering options"


class FakeBackend:
    def __init__(self, candidates):
        self._c = candidates

    def propose(self, utterance):
        return [dict(c) for c in self._c]


def texts(sqs):
    return [s.text for s in sqs]


# ------------------------------------------------------------- R1
def test_single_intent_emits_exactly_one_sub_query(make_decomposer):
    sqs = make_decomposer().decompose("What is the seated capacity of Venue B?", "p1")
    assert len(sqs) == 1


def test_single_intent_from_backend_is_not_padded(make_decomposer):
    be = FakeBackend([{"text": "seated capacity of Venue B", "intent_type": "constraint"}])
    assert len(make_decomposer(backend=be).decompose("whatever", "p1")) == 1


# ------------------------------------------------------------- R2
def test_three_intent_utterance_splits_into_three(make_decomposer):
    sqs = make_decomposer().decompose(DEMO_COMPOUND, "p1")
    assert len(sqs) == 3, texts(sqs)


def test_sub_queries_are_pairwise_non_overlapping(make_decomposer, embedding_backend):
    sqs = make_decomposer().decompose(DEMO_COMPOUND, "p1")
    vecs = embedding_backend.encode(texts(sqs))
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, 0.0)
    assert sims.max() < 0.90, sims


def test_catering_and_cancellation_intents_are_both_present(make_decomposer):
    joined = " | ".join(texts(make_decomposer().decompose(DEMO_COMPOUND, "p1"))).lower()
    assert "cancellation" in joined and "catering" in joined


# ------------------------------------------------------------- R3
def test_exact_duplicate_candidates_merge_to_one(make_decomposer):
    be = FakeBackend([
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
    ])
    assert len(make_decomposer(backend=be).decompose("x", "p1")) == 1


def test_duplicate_merge_keeps_the_first_and_still_keeps_distinct_ones(make_decomposer):
    be = FakeBackend([
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
        {"text": "catering options on site", "intent_type": "factual_lookup"},
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
    ])
    out = make_decomposer(backend=be).decompose("x", "p1")
    assert len(out) == 2 and out[0].text.startswith("cancellation")


# ------------------------------------------------------------- R4
def test_hard_cap_of_four(make_decomposer):
    be = FakeBackend([{"text": t, "intent_type": "factual_lookup"} for t in [
        "venue capacity seats", "cancellation refund deadline", "catering menu vegetarian",
        "travel reimbursement international", "parking accessibility venue", "booking surcharge weekend",
    ]])
    assert len(make_decomposer(backend=be).decompose("x", "p1")) <= 4


# ------------------------------------------------------------- R5
def test_sub_query_fields_are_well_formed(make_decomposer):
    sqs = make_decomposer().decompose(DEMO_COMPOUND, "parent-123")
    assert len({s.id for s in sqs}) == len(sqs)
    for s in sqs:
        assert s.id and s.text.strip()
        assert s.intent_type in VALID_INTENTS
        assert s.parent_utterance_id == "parent-123"


def test_heuristic_backend_intent_tags():
    h = HeuristicDecomposeBackend()
    tags = {c["text"]: c["intent_type"] for c in h.propose(
        "what is the cancellation policy and what is the venue capacity and compare the catering")}
    assert "policy" in tags.values() and "constraint" in tags.values() and "comparison" in tags.values()


# ------------------------------------------------------------- R6
def test_cap_comes_from_config(tmp_path, make_decomposer):
    cfg = tmp_path / "d.yaml"
    cfg.write_text("merge_similarity_threshold: 0.90\nmax_sub_queries: 2\n", encoding="utf-8")
    sqs = make_decomposer(config_path=str(cfg)).decompose(DEMO_COMPOUND, "p1")
    assert len(sqs) == 2


def test_merge_threshold_comes_from_config(tmp_path, make_decomposer):
    cfg = tmp_path / "d.yaml"
    cfg.write_text("merge_similarity_threshold: 0.0\nmax_sub_queries: 4\n", encoding="utf-8")
    # threshold 0.0 => everything counts as a duplicate of the first candidate
    assert len(make_decomposer(config_path=str(cfg)).decompose(DEMO_COMPOUND, "p1")) == 1


# ------------------------------------------------------------- R7
def test_decomposition_telemetry(make_decomposer, telemetry):
    make_decomposer(trace_id="T-dec").decompose(DEMO_COMPOUND, "p1")
    evs = [e for e in telemetry.events_for_trace("T-dec") if e["event"] == "decomposition_completed"]
    assert len(evs) == 1
    p = evs[0]["payload"]
    assert evs[0]["stage"] == "decomposer"
    assert p["n_raw"] >= p["n_after_merge"] >= p["n_final"] >= 1
    assert p["latency_ms"] >= 0 and len(p["sub_queries"]) == p["n_final"]


def test_telemetry_records_the_merge(make_decomposer, telemetry):
    be = FakeBackend([{"text": "same text here", "intent_type": "policy"}] * 3)
    make_decomposer(backend=be, trace_id="T-merge").decompose("x", "p1")
    p = telemetry.events_for_trace("T-merge")[0]["payload"]
    assert (p["n_raw"], p["n_after_merge"], p["n_final"]) == (3, 1, 1)


# ------------------------------------------------------------- R8
UTT = "I need the cancellation policy and the catering options"


def test_llm_bad_json_falls_back_to_whole_utterance():
    out = GeminiDecomposeBackend(client=FakeLLMClient(exc=ValueError("bad json"))).propose(UTT)
    assert [c["text"] for c in out] == [UTT]


def test_llm_empty_list_falls_back_to_whole_utterance():
    out = GeminiDecomposeBackend(client=FakeLLMClient(response={"sub_queries": []})).propose(UTT)
    assert [c["text"] for c in out] == [UTT]


def test_llm_auth_error_is_surfaced():
    with pytest.raises(RuntimeError):
        GeminiDecomposeBackend(client=FakeLLMClient(exc=RuntimeError("401"))).propose(UTT)


def test_llm_outage_degrades_to_heuristic_split():
    out = GeminiDecomposeBackend(client=FakeLLMClient(exc=TimeoutError("down"))).propose(UTT)
    assert len(out) == 2


def test_llm_valid_response_passes_through():
    resp = {"sub_queries": [{"text": "cancellation policy", "intent_type": "policy"},
                            {"text": "catering options", "intent_type": "factual_lookup"}]}
    assert GeminiDecomposeBackend(client=FakeLLMClient(response=resp)).propose(UTT) == resp["sub_queries"]


# ------------------------------------------------------------- R9
def test_every_sub_query_retrieves_its_own_evidence(make_decomposer, retriever):
    sqs = make_decomposer().decompose(DEMO_COMPOUND, "p1")
    ev = retriever.search(sqs)
    covered = {c.sub_query_id for c in ev}
    assert {s.id for s in sqs} <= covered


def test_cancellation_sub_query_hits_the_cancellation_doc(make_decomposer, retriever):
    sqs = make_decomposer().decompose(DEMO_COMPOUND, "p1")
    cancel = next(s for s in sqs if "cancellation" in s.text.lower())
    docs = [c.doc_id for c in retriever.search([cancel])[:3]]
    assert "Doc_02" in docs


# ------------------------------------------------------------- R10
def _backend_from_env():
    backend_choice = os.getenv("DECOMPOSER_BACKEND", "heuristic").lower()
    if backend_choice == "groq":
        from decomposer.backends import GroqDecomposeBackend
        return GroqDecomposeBackend(), 0.70, "groq"
    if backend_choice == "gemini":
        return GeminiDecomposeBackend(), 0.70, "gemini"
    if backend_choice == "openai":
        from decomposer.backends import OpenAIDecomposeBackend
        return OpenAIDecomposeBackend(), 0.70, "openai"
    return HeuristicDecomposeBackend(), 0.50, "heuristic (plumbing threshold only)"


def test_g3_sub_intent_count_accuracy_on_heldout(make_decomposer):
    backend, threshold, label = _backend_from_env()
    decomposer = make_decomposer(backend=backend)
    held = heldout(COMPOUND)
    rows = [(u, gold, len(decomposer.decompose(u, "p"))) for u, gold in held]
    acc = sum(g == n for _, g, n in rows) / len(rows)
    print(f"\nG3 [{label}] held-out accuracy {acc:.0%} ({len(rows)} items); wrong: "
          f"{[(u[:40], g, n) for u, g, n in rows if g != n]}")
    assert acc >= threshold


def test_g3_full_set_report(make_decomposer):
    # Report-only on ALL items (tuning view). Never quote this as the final G3.
    backend, _, label = _backend_from_env()
    decomposer = make_decomposer(backend=backend)
    wrong = [(u[:50], g, len(decomposer.decompose(u, "p"))) for u, g in COMPOUND]
    wrong = [w for w in wrong if w[1] != w[2]]
    print(f"\n[{label}] mismatches on full set: {wrong}")


@pytest.mark.xfail(reason="Heuristic splits on every 'and'; 'check-in and check-out' is ONE intent. "
                          "Good Step 9 'entangled compound' edge case.", strict=False)
def test_entangled_single_intent_is_not_over_split(make_decomposer):
    assert len(make_decomposer().decompose("What are the check-in and check-out rules for Venue B", "p")) == 1


def _has_groq_key() -> bool:
    if os.getenv("GROQ_API_KEY"):
        return True
    try:
        from llm.groq_client import _load_env_file_if_needed
        _load_env_file_if_needed()
        return bool(os.getenv("GROQ_API_KEY"))
    except Exception:
        return False


@pytest.mark.skipif(not _has_groq_key(), reason="needs GROQ_API_KEY (real Groq LLM)")
def test_groq_llm_resolves_context_into_each_sub_query(make_decomposer):
    try:
        from decomposer.backends import GroqDecomposeBackend
        be = GroqDecomposeBackend()
        sqs = make_decomposer(backend=be).decompose(
            "I am planning a workshop in Pune. What is the cancellation policy for it and is catering available there", "p")
    except RuntimeError as e:
        if "rejected the API key" in str(e) or "No Groq API key found" in str(e):
            pytest.skip(f"GROQ_API_KEY invalid or missing: {e}")
        raise
    assert len(sqs) == 2
    assert all(re.search(r"pune", s.text, re.I) for s in sqs), texts(sqs)


@pytest.mark.skipif(not _has_groq_key(), reason="needs GROQ_API_KEY (real Groq LLM)")
def test_groq_llm_resolves_mid_utterance_speech_correction(make_decomposer):
    try:
        from decomposer.backends import GroqDecomposeBackend
        be = GroqDecomposeBackend()
        sqs = make_decomposer(backend=be).decompose(
            "Book a venue in Pune... actually Mumbai for 30 people", "p")
    except RuntimeError as e:
        if "rejected the API key" in str(e) or "No Groq API key found" in str(e):
            pytest.skip(f"GROQ_API_KEY invalid or missing: {e}")
        raise
    assert any("mumbai" in s.text.lower() for s in sqs)
    assert not any("pune" in s.text.lower() for s in sqs)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY (real LLM)")
def test_real_llm_resolves_context_into_each_sub_query(make_decomposer):
    try:
        be = GeminiDecomposeBackend()
        sqs = make_decomposer(backend=be).decompose(
            "I am planning a workshop in Pune. What is the cancellation policy for it and is catering available there", "p")
    except RuntimeError as e:
        if "rejected the API key" in str(e):
            pytest.skip(f"GEMINI_API_KEY invalid or expired: {e}")
        raise
    assert len(sqs) == 2
    assert all(re.search(r"pune", s.text, re.I) for s in sqs), texts(sqs)


@pytest.mark.skipif(not os.getenv("RUN_OPENAI_TESTS"), reason="OpenAI tests opt-in; user switched to Groq")
def test_openai_llm_resolves_context_into_each_sub_query(make_decomposer):
    from decomposer.backends import OpenAIDecomposeBackend
    be = OpenAIDecomposeBackend()
    sqs = make_decomposer(backend=be).decompose(
        "I am planning a workshop in Pune. What is the cancellation policy for it and is catering available there", "p")
    assert len(sqs) == 2
    assert all(re.search(r"pune", s.text, re.I) for s in sqs), texts(sqs)


@pytest.mark.skipif(not os.getenv("RUN_OPENAI_TESTS"), reason="OpenAI tests opt-in; user switched to Groq")
def test_openai_llm_resolves_mid_utterance_speech_correction(make_decomposer):
    from decomposer.backends import OpenAIDecomposeBackend
    be = OpenAIDecomposeBackend()
    sqs = make_decomposer(backend=be).decompose(
        "Book a venue in Pune... actually Mumbai for 30 people", "p")
    assert any("mumbai" in s.text.lower() for s in sqs)
    assert not any("pune" in s.text.lower() for s in sqs)

