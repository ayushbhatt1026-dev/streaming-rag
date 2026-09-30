"""
M1, Stage 2 — the constrained-output LLM confirm. Only called for
utterances that survived the Stage 1 rule gate (i.e. already look
stable-ish) — this is what makes it affordable to call an LLM here at all.

Two implementations, same interface (`confirm`):
  - GeminiStage2Confirm: production. Requires GEMINI_API_KEY.
  - HeuristicStage2Confirm: OFFLINE DEV FALLBACK — deterministic rules, no
    network. Use this to build/test the rest of the pipeline (decomposer,
    retriever, synthesizer, telemetry) without burning API calls or
    needing a key yet. Swap to Gemini before you measure real G2 numbers —
    the heuristic is intentionally simple and will not match real semantic
    judgement on ambiguous cases.
"""

from __future__ import annotations

from typing import Protocol

from contracts import ControllerDecision, Decision
from llm.gemini_client import GeminiLLMClient

_SYSTEM_PROMPT = """You are the retrieval-timing controller for a streaming RAG system.
Given the transcript of an utterance still being spoken (or just finished), decide one of:
  RETRIEVE_PROVISIONAL - intent is stable enough to search early, but the utterance may still be going
  RETRIEVE_FINAL - the utterance is complete and ready for authoritative search
  SUPPRESS - this is a formatting/style/meta request needing no new search
  WAIT - not enough information yet to search meaningfully

Respond ONLY with JSON: {"decision": "...", "confidence": 0.0-1.0, "reason": "...", "is_refinement_of_prior_answer": false}
"""


class Stage2Confirm(Protocol):
    def confirm(self, utterance_so_far: str, is_final: bool) -> ControllerDecision: ...


class GeminiStage2Confirm:
    def __init__(self, client: GeminiLLMClient | None = None) -> None:
        self._client = client or GeminiLLMClient()

    def confirm(self, utterance_so_far: str, is_final: bool) -> ControllerDecision:
        user_prompt = (
            f"Utterance so far: \"{utterance_so_far}\"\n"
            f"Is this the final chunk of the utterance? {is_final}"
        )
        try:
            data = self._client.complete_json(_SYSTEM_PROMPT, user_prompt)
            return ControllerDecision(
                decision=Decision(data["decision"]),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
                is_refinement_of_prior_answer=bool(data.get("is_refinement_of_prior_answer", False)),
            )
        except (ValueError, KeyError) as e:
            # Fail safe: an unparseable/invalid LLM response should never
            # crash the turn or (worse) silently fire a bad retrieval —
            # WAIT is always the safe default.
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 parse failure, defaulting to WAIT: {e}",
            )


class HeuristicStage2Confirm:
    """OFFLINE DEV FALLBACK. See module docstring."""

    def confirm(self, utterance_so_far: str, is_final: bool) -> ControllerDecision:
        word_count = len(utterance_so_far.split())
        if is_final:
            return ControllerDecision(
                decision=Decision.RETRIEVE_FINAL,
                confidence=0.9,
                reason="heuristic: is_final flag set",
            )
        # Stable-enough-to-escalate already implied by reaching Stage 2 at
        # all (the rule gate only escalates on entity stability). Fire a
        # provisional retrieval, with confidence scaled by how much text
        # we've seen so far.
        confidence = min(0.85, 0.4 + 0.05 * word_count)
        return ControllerDecision(
            decision=Decision.RETRIEVE_PROVISIONAL,
            confidence=confidence,
            reason="heuristic: entities stable, utterance still forming",
        )
