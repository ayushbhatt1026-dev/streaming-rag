"""
Step 4 — walking skeleton. Requirements covered (blueprint Phase 1, Step 4):

  R1  simulator replays a sentence as word-timed chunks, deterministically
  R2  telemetry bus writes JSONL; every stage emits events with ONE trace_id
  R3  retrieval returns EvidenceChunks with usable citation metadata
  R4  one simulated utterance -> cited answer (the "walking skeleton" exit test)
  R5  complete trace: chunk_received ... turn_completed
  R6  a turn never silently produces nothing
  (Clean `docker compose up` run is manual — see the checklist in the chat.)

Run: python -m pytest tests/test_step4_skeleton.py -v
"""

import json
import re
from dataclasses import FrozenInstanceError

import pytest

from contracts import (
    AnswerVersion, Claim, ClaimStatus, ControllerDecision, Decision,
    EvidenceChunk, SubQuery, TelemetryEvent, TranscriptChunk,
)
from simulator.transcript_simulator import simulate_utterance
from telemetry.bus import TelemetryBus

CITATION_RE = re.compile(r"\[Doc_\d+ \u00a7[^\]]+\]")
COMPOUND = (
    "I need to plan a customer workshop in Pune for 30 people and I need the "
    "cancellation policy and the catering options"
)


# ---------------------------------------------------------------- contracts
def test_all_seven_contracts_instantiate():
    TranscriptChunk(session_id="s", seq=0, t_ms=0.0, text_delta="hi")
    ControllerDecision(decision=Decision.WAIT, confidence=0.1, reason="r")
    SubQuery(id="q", text="t")
    ec = EvidenceChunk(chunk_id="c", doc_id="Doc_01", section_ref="Section 1",
                       heading_path="h", text="x")
    claim = Claim(id="c1", text="x", citations=[ec.citation()])
    AnswerVersion(version=1, claims=[claim])
    TelemetryEvent(trace_id="t", session_id="s", stage="a", event="b", t_ms=0.0)


def test_transcript_chunk_is_immutable():
    tc = TranscriptChunk(session_id="s", seq=0, t_ms=0.0, text_delta="hi")
    with pytest.raises(FrozenInstanceError):
        tc.seq = 5


def test_citation_label_format():
    ec = EvidenceChunk(chunk_id="c", doc_id="Doc_01", section_ref="Section 1",
                       heading_path="h", text="x")
    assert ec.citation() == "[Doc_01 \u00a7Section 1]"


def test_answer_version_citations_skip_retracted():
    a = Claim(id="c1", text="a", citations=["[Doc_01 \u00a7A]"])
    b = Claim(id="c2", text="b", citations=["[Doc_02 \u00a7B]"], status=ClaimStatus.RETRACTED)
    assert AnswerVersion(version=1, claims=[a, b]).citations_all == ["[Doc_01 \u00a7A]"]


# ---------------------------------------------------------------- simulator
def test_simulator_chunks_reassemble_to_original_text():
    text = "I need to plan a workshop in Pune"
    chunks = simulate_utterance(text, session_id="s")
    assert " ".join(c.text_delta for c in chunks) == text


def test_simulator_final_flag_only_on_last_chunk_and_seq_monotonic():
    chunks = simulate_utterance("one two three four five six seven", session_id="s")
    assert [c.is_final for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_simulator_default_is_two_words_per_chunk():
    chunks = simulate_utterance("a b c d e f", session_id="s")
    assert len(chunks) == 3
    assert all(len(c.text_delta.split()) == 2 for c in chunks)


def test_simulator_timing_is_increasing_and_realistic():
    chunks = simulate_utterance(COMPOUND, session_id="s")
    times = [c.t_ms for c in chunks]
    assert times == sorted(times)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(200 <= g <= 400 for g in gaps), gaps


def test_simulator_is_deterministic_with_default_seed():
    a = simulate_utterance(COMPOUND, session_id="s")
    b = simulate_utterance(COMPOUND, session_id="s")
    assert a == b


# ---------------------------------------------------------------- telemetry
def test_telemetry_writes_one_valid_json_line_per_event(tmp_path):
    bus = TelemetryBus(jsonl_path=str(tmp_path / "t.jsonl"))
    for i in range(3):
        bus.emit(TelemetryEvent(trace_id="T1", session_id="s", stage="controller",
                                event="controller_decision", t_ms=float(i), payload={"i": i}))
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        ev = json.loads(line)
        assert {"trace_id", "session_id", "stage", "event", "t_ms", "payload"} <= set(ev)


def test_telemetry_events_for_trace_filters_by_trace_id(tmp_path):
    bus = TelemetryBus(jsonl_path=str(tmp_path / "t.jsonl"))
    for tid in ("A", "B", "A"):
        bus.emit(TelemetryEvent(trace_id=tid, session_id="s", stage="x", event="e", t_ms=0.0))
    assert len(bus.events_for_trace("A")) == 2
    assert len(bus.events_for_trace("B")) == 1
    assert bus.events_for_trace("nope") == []


# ---------------------------------------------------------------- retrieval
@pytest.mark.parametrize("query,expected_doc", [
    ("What is the seated capacity of Venue B", "Doc_01"),
    ("cancellation policy for a workshop", "Doc_02"),
    ("catering options", "Doc_03"),
    ("international travel reimbursement approval", "Doc_04"),
])
def test_retriever_finds_the_right_document(retriever, query, expected_doc):
    ev = retriever.search([SubQuery(id="q1", text=query)])
    assert expected_doc in [c.doc_id for c in ev[:3]], [c.doc_id for c in ev]


def test_retriever_evidence_has_citation_metadata(retriever):
    ev = retriever.search([SubQuery(id="q1", text="cancellation policy")])
    assert ev
    for c in ev:
        assert c.doc_id and c.section_ref and c.text.strip()
        assert CITATION_RE.fullmatch(c.citation()), c.citation()
        assert c.sub_query_id == "q1"


# ------------------------------------------------------------- end-to-end
def test_skeleton_produces_cited_answer(run_turn):
    turn = run_turn(COMPOUND)
    assert turn.result.answer.strip()
    assert CITATION_RE.search(turn.result.answer), turn.result.answer


def test_every_citation_in_answer_was_actually_retrieved(run_turn):
    turn = run_turn(COMPOUND)
    retrieved = {c.citation() for c in turn.result.evidence}
    cited = set(CITATION_RE.findall(turn.result.answer))
    assert cited and cited <= retrieved


def test_skeleton_answer_has_one_line_per_sub_query(run_turn):
    turn = run_turn(COMPOUND)
    assert len(turn.result.sub_query_texts) >= 2
    assert len(turn.result.answer.splitlines()) == len(turn.result.sub_query_texts)


def test_full_trace_covers_every_stage_with_one_trace_id(run_turn):
    turn = run_turn(COMPOUND)
    events = turn.events()
    stages = {e["stage"] for e in events}
    names = {e["event"] for e in events}
    assert {"stream_buffer", "controller", "decomposer", "retrieval", "synthesis"} <= stages
    assert {"chunk_received", "controller_decision", "decomposition_completed",
            "retrieval_started", "retrieval_completed", "turn_completed"} <= names
    assert all(e["trace_id"] == turn.trace_id for e in events)


def test_one_chunk_received_event_per_chunk_up_to_the_stop_point(run_turn):
    turn = run_turn(COMPOUND)
    n_chunks_in_utterance = len(simulate_utterance(COMPOUND, session_id="x"))
    assert len(turn.events_named("chunk_received")) == n_chunks_in_utterance
    assert len(turn.events_named("controller_decision")) == n_chunks_in_utterance


def test_turn_completed_is_emitted_exactly_once(run_turn):
    assert len(run_turn(COMPOUND).events_named("turn_completed")) == 1


def test_two_turns_get_separate_traces(run_turn):
    t1 = run_turn("What is the seated capacity of Venue B?")
    t2 = run_turn("Is there a weekend booking surcharge")
    assert t1.trace_id != t2.trace_id
    assert t1.events() and t2.events()
    assert not ({e["trace_id"] for e in t1.events()} & {e["trace_id"] for e in t2.events()})


def test_suppressed_turn_still_completes_with_zero_retrievals(run_turn):
    turn = run_turn("Can you repeat that in two bullets please")
    assert turn.result.final_decision == Decision.SUPPRESS
    assert turn.result.n_retrievals_fired == 0
    assert turn.events_named("retrieval_started") == []
    assert len(turn.events_named("turn_completed")) == 1


def test_turn_never_silently_produces_nothing(run_turn):
    for utt in ["What is the seated capacity of Venue B?", "ok", "Is there a weekend booking surcharge"]:
        assert run_turn(utt).result.answer.strip(), utt


def test_unanswerable_question_still_returns_an_answer_string(run_turn):
    # Honest-uncertainty is Step 8; at Step 4 we only require no crash + non-empty.
    turn = run_turn("Is there a weekend booking surcharge")
    assert turn.result.answer
