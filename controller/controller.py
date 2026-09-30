"""
M1 — RetrievalController. Given each incoming TranscriptChunk, appends it
to the session's buffer, runs Stage 1, escalates to Stage 2 if needed, and
emits a `controller_decision` telemetry event either way (needed for G2/G6).

Note on scope (Phase 1-2, Steps 4-6): REFINE is part of the Decision
taxonomy in contracts.py but this controller never emits it yet — REFINE
only becomes meaningful once a prior AnswerVersion exists to refine against
(M4, the Refinement Engine — that's a later step). Wiring it in now would
be a decision this pipeline can't act on.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from contracts import ControllerDecision, Decision, TelemetryEvent, TranscriptChunk
from controller.rule_gate import RuleGate
from controller.stage2 import Stage2Confirm
from telemetry.bus import TelemetryBus


def _load_config(path: str = "config/controller.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RetrievalController:
    stage2: Stage2Confirm
    telemetry: TelemetryBus
    session_id: str
    trace_id: str
    config_path: str = "config/controller.yaml"

    def __post_init__(self) -> None:
        cfg = _load_config(self.config_path)
        self._rule_gate = RuleGate(
            token_floor=cfg["token_floor"],
            stability_window=cfg["stability_window"],
        )
        self._always_escalate_on_final = cfg["always_escalate_on_final"]
        self._buffer_text = ""
        self._expected_final_words = 25  # rough heuristic for progress_pct; see note below

    def decide(self, chunk: TranscriptChunk) -> ControllerDecision:
        start = time.perf_counter()

        self._buffer_text = (self._buffer_text + " " + chunk.text_delta).strip()

        rule_result = self._rule_gate.evaluate(self._buffer_text)
        if rule_result is not None and not (chunk.is_final and self._always_escalate_on_final):
            decision = ControllerDecision(
                decision=rule_result, confidence=1.0, reason="stage1_rule_gate",
            )
        else:
            decision = self.stage2.confirm(self._buffer_text, chunk.is_final)

        latency_ms = (time.perf_counter() - start) * 1000

        # utterance_progress_pct is a rough heuristic (word count vs. an
        # assumed typical utterance length), not a real ASR end-of-speech
        # probability. Good enough to show a progress trend in telemetry;
        # don't over-interpret the absolute number.
        progress_pct = 100.0 if chunk.is_final else min(
            95.0, 100.0 * len(self._buffer_text.split()) / self._expected_final_words
        )

        self.telemetry.emit(
            TelemetryEvent(
                trace_id=self.trace_id,
                session_id=self.session_id,
                stage="controller",
                event="controller_decision",
                t_ms=chunk.t_ms,
                payload={
                    "decision": decision.decision.value,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "latency_ms": round(latency_ms, 3),
                    "utterance_progress_pct": round(progress_pct, 1),
                    "buffer_text": self._buffer_text,
                },
            )
        )
        return decision

    @property
    def utterance_so_far(self) -> str:
        return self._buffer_text
