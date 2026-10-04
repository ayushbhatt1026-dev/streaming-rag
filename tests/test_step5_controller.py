"""
Step 5 — early retrieval / Retrieval Controller (Gate G2). Requirements:

  R1  Stage 1 rule gate: token floor, no entity, dangling connector -> WAIT;
      presentation-only -> SUPPRESS; stable entities -> escalate to Stage 2
  R2  Stage 2: constrained output {decision, confidence, reason, refinement flag};
      fails SAFE (parse failure -> WAIT, transient outage -> WAIT / heuristic on final)
  R3  Stage 1 short-circuits: Stage 2 is NOT called for obvious cases
  R4  thresholds live in config/controller.yaml, not in code
  R5  controller_decision telemetry has latency_ms + utterance_progress_pct
  R6  retrieval starts BEFORE is_final on eligible utterances (>=80%)
  R7  presentation-only turn -> zero retrievals; incomplete turns never retrieve
  R8  mid-utterance correction ("Pune ... actually Mumbai") re-issues retrieval
  R9  false-trigger rate measured against a no-retrieval set

Run: python -m pytest tests/test_step5_controller.py -v -s
"""

import dataclasses

import pytest
import yaml

from conftest import FakeLLMClient
from contracts import ControllerDecision, Decision
from controller.stage2 import (
    GeminiStage2Confirm,
    GroqStage2Confirm,
    HeuristicStage2Confirm,
    OpenAIStage2Confirm,
)
from labelled_sets import (
    EARLY_ELIGIBLE, INCOMPLETE, PRESENTATION_ONLY, REAL_QUESTIONS, SUMMARIZE_VARIANTS,
)
from simulator.transcript_simulator import simulate_utterance

RETRIEVE = (Decision.RETRIEVE_PROVISIONAL, Decision.RETRIEVE_FINAL)


class SpyStage2:
    def __init__(self):
        self.calls = []
        self._inner = HeuristicStage2Confirm()

    def confirm(self, utterance_so_far, is_final, **kw):
        self.calls.append((utterance_so_far, is_final))
        return self._inner.confirm(utterance_so_far, is_final, **kw)


def replay(controller, text, final_on_last=True):
    out = []
    for ch in simulate_utterance(text, session_id="s"):
        if not final_on_last:
            ch = dataclasses.replace(ch, is_final=False)
        out.append(controller.decide(ch))
    return out


# ------------------------------------------------------------- R1 / R3
def test_stage1_handles_presentation_only_without_calling_stage2(make_controller):
    spy = SpyStage2()
    decisions = replay(make_controller(stage2=spy), "Can you repeat that in two bullets", final_on_last=False)
    assert Decision.SUPPRESS in [d.decision for d in decisions]
    assert spy.calls == []


def test_stage1_handles_too_short_without_calling_stage2(make_controller):
    spy = SpyStage2()
    decisions = replay(make_controller(stage2=spy), "I need to", final_on_last=False)
    assert all(d.decision == Decision.WAIT for d in decisions)
    assert spy.calls == []


def test_stage2_is_only_called_for_the_ambiguous_remainder(make_controller):
    spy = SpyStage2()
    utt = EARLY_ELIGIBLE[0]
    n_chunks = len(simulate_utterance(utt, session_id="s"))
    replay(make_controller(stage2=spy), utt)
    assert 1 <= len(spy.calls) < n_chunks


def test_stage1_decisions_have_confidence_one_and_rule_gate_reason(make_controller):
    d = make_controller().decide(simulate_utterance("I need", session_id="s")[0])
    assert d.decision == Decision.WAIT and d.confidence == 1.0 and d.reason == "stage1_rule_gate"


def test_final_chunk_always_reaches_stage2(make_controller):
    spy = SpyStage2()
    replay(make_controller(stage2=spy), "ok thanks")  # tiny utterance, still must get a final decision
    assert spy.calls and spy.calls[-1][1] is True


def test_trailing_conjunction_waits(make_controller):
    decisions = replay(make_controller(), INCOMPLETE[-1], final_on_last=False)
    assert decisions[-1].decision == Decision.WAIT
    assert not any(d.decision in RETRIEVE for d in decisions)


# ------------------------------------------------------------- R2
def _gemini(client):
    return GeminiStage2Confirm(client=client)


def _openai(client):
    return OpenAIStage2Confirm(client=client)


def _groq(client):
    return GroqStage2Confirm(client=client)


def test_stage2_valid_json_becomes_controller_decision():
    c = FakeLLMClient(response={"decision": "RETRIEVE_PROVISIONAL", "confidence": 0.8,
                                "reason": "stable", "is_refinement_of_prior_answer": False})
    d = _gemini(c).confirm("I need a venue in Pune for 30 people", False)
    assert isinstance(d, ControllerDecision)
    assert (d.decision, d.confidence, d.reason) == (Decision.RETRIEVE_PROVISIONAL, 0.8, "stable")
    assert d.is_refinement_of_prior_answer is False

    d_openai = _openai(c).confirm("I need a venue in Pune for 30 people", False)
    assert isinstance(d_openai, ControllerDecision)
    assert (d_openai.decision, d_openai.confidence, d_openai.reason) == (Decision.RETRIEVE_PROVISIONAL, 0.8, "stable")

    d_groq = _groq(c).confirm("I need a venue in Pune for 30 people", False)
    assert isinstance(d_groq, ControllerDecision)
    assert (d_groq.decision, d_groq.confidence, d_groq.reason) == (Decision.RETRIEVE_PROVISIONAL, 0.8, "stable")


def test_stage2_invalid_decision_string_fails_safe_to_wait():
    d = _gemini(FakeLLMClient(response={"decision": "MAYBE"})).confirm("x y z", False)
    assert d.decision == Decision.WAIT and d.confidence == 0.0


def test_stage2_missing_keys_fails_safe_to_wait():
    assert _gemini(FakeLLMClient(response={})).confirm("x", False).decision == Decision.WAIT


def test_stage2_bad_json_valueerror_fails_safe_to_wait():
    assert _gemini(FakeLLMClient(exc=ValueError("not json"))).confirm("x", False).decision == Decision.WAIT


def test_stage2_auth_error_is_surfaced_not_swallowed():
    with pytest.raises(RuntimeError):
        _gemini(FakeLLMClient(exc=RuntimeError("401"))).confirm("x", False)


def test_stage2_transient_outage_midstream_waits():
    d = _gemini(FakeLLMClient(exc=TimeoutError("boom"))).confirm("x y", False)
    assert d.decision == Decision.WAIT


def test_stage2_transient_outage_on_final_falls_back_to_heuristic_retrieve():
    d = _gemini(FakeLLMClient(exc=TimeoutError("boom"))).confirm("x y", True)
    assert d.decision == Decision.RETRIEVE_FINAL and "fallback" in d.reason


def test_heuristic_stage2_shapes():
    h = HeuristicStage2Confirm()
    assert h.confirm("a b c", True).decision == Decision.RETRIEVE_FINAL
    prov = h.confirm("a b c", False)
    assert prov.decision == Decision.RETRIEVE_PROVISIONAL and 0.0 <= prov.confidence <= 1.0


# ------------------------------------------------------------- R4
def test_config_has_required_keys_and_sane_types():
    from conftest import CONTROLLER_CFG
    cfg = yaml.safe_load(open(CONTROLLER_CFG, encoding="utf-8"))
    assert isinstance(cfg["token_floor"], int) and cfg["token_floor"] > 0
    assert isinstance(cfg["stability_window"], int) and cfg["stability_window"] >= 1
    assert isinstance(cfg["always_escalate_on_final"], bool)


def test_thresholds_really_come_from_config(tmp_path, make_controller):
    strict = tmp_path / "strict.yaml"
    strict.write_text("token_floor: 99\nstability_window: 2\nalways_escalate_on_final: false\n", encoding="utf-8")
    never = replay(make_controller(config_path=str(strict)), EARLY_ELIGIBLE[0])
    assert all(d.decision == Decision.WAIT for d in never)
    normal = replay(make_controller(), EARLY_ELIGIBLE[0])
    assert any(d.decision in RETRIEVE for d in normal)


def test_always_escalate_on_final_flag_is_respected(tmp_path, make_controller):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("token_floor: 99\nstability_window: 2\nalways_escalate_on_final: true\n", encoding="utf-8")
    decisions = replay(make_controller(config_path=str(cfg)), "What is the seated capacity of Venue B?")
    assert decisions[-1].decision == Decision.RETRIEVE_FINAL


# ------------------------------------------------------------- R5
def test_controller_decision_telemetry_payload(make_controller, telemetry):
    ctl = make_controller(trace_id="T-payload")
    replay(ctl, EARLY_ELIGIBLE[0])
    events = [e for e in telemetry.events_for_trace("T-payload") if e["event"] == "controller_decision"]
    assert events
    for e in events:
        assert e["stage"] == "controller"
        assert {"decision", "confidence", "reason", "latency_ms", "utterance_progress_pct"} <= set(e["payload"])
        assert e["payload"]["decision"] in {d.value for d in Decision}


def test_progress_pct_is_monotonic_and_100_on_final(make_controller, telemetry):
    replay(make_controller(trace_id="T-prog"), EARLY_ELIGIBLE[1])
    pct = [e["payload"]["utterance_progress_pct"]
           for e in telemetry.events_for_trace("T-prog") if e["event"] == "controller_decision"]
    assert pct == sorted(pct) and pct[-1] == 100.0 and all(p < 100.0 for p in pct[:-1])


def test_stage1_latency_is_well_under_the_50ms_budget(make_controller, telemetry):
    replay(make_controller(trace_id="T-lat"), "Can you repeat that in two bullets", final_on_last=False)
    replay(make_controller(trace_id="T-lat"), "I need to", final_on_last=False)
    lat = [e["payload"]["latency_ms"] for e in telemetry.events_for_trace("T-lat")
           if e["event"] == "controller_decision" and e["payload"]["reason"] == "stage1_rule_gate"]
    assert lat and max(lat) < 50.0, lat


# ------------------------------------------------------------- R6
def test_early_retrieval_happens_before_final_chunk(run_turn):
    turn = run_turn(EARLY_ELIGIBLE[0])
    assert turn.result.early_retrieval is True
    first = turn.events_named("retrieval_started")[0]
    assert first["payload"]["trigger"] == "provisional"
    decisions = turn.events_named("controller_decision")
    prov = [e for e in decisions if e["payload"]["decision"] == "RETRIEVE_PROVISIONAL"]
    assert prov and prov[0]["payload"]["utterance_progress_pct"] < 100.0


def test_g2_early_retrieval_rate_on_eligible_set(run_turn):
    hits = [run_turn(u).result.early_retrieval for u in EARLY_ELIGIBLE]
    rate = sum(hits) / len(hits)
    print(f"\nG2 early-retrieval rate: {rate:.0%} ({sum(hits)}/{len(hits)})")
    missed = [u for u, h in zip(EARLY_ELIGIBLE, hits) if not h]
    assert rate >= 0.80, f"G2 below 80%. Missed: {missed}"


def test_final_decision_on_a_full_utterance_is_retrieve_final(run_turn):
    assert run_turn(EARLY_ELIGIBLE[0]).result.final_decision == Decision.RETRIEVE_FINAL


# ------------------------------------------------------------- R8
def test_mid_utterance_correction_reissues_retrieval_with_corrected_text(run_turn):
    utt = ("I need to plan a customer workshop in Pune for 30 people in the city centre "
           "actually Mumbai instead")
    turn = run_turn(utt)
    starts = turn.events_named("retrieval_started")
    assert len(starts) >= 2, "expected a provisional retrieval AND a post-correction retrieval"
    assert "mumbai" not in starts[0]["payload"]["utterance_so_far"].lower()
    assert "mumbai" in starts[-1]["payload"]["utterance_so_far"].lower()
    assert starts[0]["payload"]["trigger"] == "provisional"
    assert starts[-1]["payload"]["trigger"] == "final"


# ------------------------------------------------------------- R7 / R9
@pytest.mark.parametrize("utt", PRESENTATION_ONLY)
def test_presentation_only_turn_fires_zero_retrievals(run_turn, utt):
    turn = run_turn(utt)
    assert turn.result.final_decision == Decision.SUPPRESS
    assert turn.result.n_retrievals_fired == 0
    assert turn.events_named("retrieval_started") == []


@pytest.mark.parametrize("utt", SUMMARIZE_VARIANTS)
def test_summarize_requests_are_suppressed(run_turn, utt):
    # EXPECTED TO FAIL TODAY: controller/rule_gate.py has r"...summariz...\b"
    # and \b can never match after "summariz" in "summarize". Fix:
    #   summariz  ->  summari[sz]\w*
    turn = run_turn(utt)
    assert turn.result.final_decision == Decision.SUPPRESS, "rule_gate 'summariz' regex bug"


@pytest.mark.parametrize("utt", INCOMPLETE)
def test_incomplete_utterances_never_trigger_retrieval(make_controller, utt):
    decisions = replay(make_controller(), utt, final_on_last=False)
    assert not any(d.decision in RETRIEVE for d in decisions), [d.decision for d in decisions]


@pytest.mark.parametrize("utt", REAL_QUESTIONS)
def test_real_questions_are_not_suppressed_and_do_retrieve(run_turn, utt):
    turn = run_turn(utt)
    assert turn.result.final_decision != Decision.SUPPRESS
    assert turn.result.n_retrievals_fired >= 1


def test_false_trigger_rate_on_no_retrieval_set(make_controller, run_turn):
    total = triggered = 0
    for utt in INCOMPLETE:
        total += 1
        ds = replay(make_controller(), utt, final_on_last=False)
        triggered += any(d.decision in RETRIEVE for d in ds)
    for utt in PRESENTATION_ONLY:
        total += 1
        triggered += run_turn(utt).result.n_retrievals_fired > 0
    rate = triggered / total
    print(f"\nFalse-trigger rate: {rate:.0%} ({triggered}/{total})  <- report this, don't chase it blindly")
    assert rate == 0.0


@pytest.mark.xfail(reason="Known limitation: keyword suppression regex also fires on real questions "
                          "containing 'shorter'/'again'/'bullet'. Document in the Step 9 edge-case report.",
                   strict=False)
def test_real_question_containing_a_suppression_word_is_not_suppressed(run_turn):
    turn = run_turn("Is there a shorter route from the airport to Venue B")
    assert turn.result.final_decision != Decision.SUPPRESS
