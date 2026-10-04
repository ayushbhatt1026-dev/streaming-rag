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

import re
from typing import Optional, Protocol

from contracts import ControllerDecision, Decision
from llm.gemini_client import GeminiLLMClient

_SYSTEM_PROMPT = """You are the retrieval-timing controller for a streaming RAG system.
Given the transcript of an utterance still being spoken (or just finished), decide one of:
  RETRIEVE_PROVISIONAL - intent is stable enough to search early, but the utterance may still be going
  RETRIEVE_FINAL - the utterance is complete and ready for authoritative search
  SUPPRESS - this is a formatting/style/meta request needing no new search
  WAIT - not enough information yet to search meaningfully

GUIDELINES:
- When is_final is False: If the utterance so far contains key entities and enough subject matter to begin searching (e.g., event type, city/venue, headcount, or topic), choose RETRIEVE_PROVISIONAL. Only choose WAIT if the partial text is completely uninformative or dangling mid-clause.
- When is_final is True: Choose RETRIEVE_FINAL if any factual search is needed, or SUPPRESS if it is formatting-only.

Respond ONLY with JSON: {"decision": "...", "confidence": 0.0-1.0, "reason": "...", "is_refinement_of_prior_answer": false}
"""


class Stage2Confirm(Protocol):
    def confirm(
        self, utterance_so_far: str, is_final: bool, prior_context: Optional[str] = None
    ) -> ControllerDecision: ...


class GeminiStage2Confirm:
    def __init__(self, client: GeminiLLMClient | None = None) -> None:
        self._client = client or GeminiLLMClient()

    def confirm(
        self, utterance_so_far: str, is_final: bool, prior_context: Optional[str] = None
    ) -> ControllerDecision:
        user_prompt = (
            f"Utterance so far: \"{utterance_so_far}\"\n"
            f"Is this the final chunk of the utterance? {is_final}"
        )
        if prior_context:
            # Step 7: the session already has an answer. Let the model say whether
            # this utterance PATCHES it (-> REFINE) or starts a new question.
            user_prompt += (
                f"\nThe user already received an answer about: {prior_context}\n"
                "Set is_refinement_of_prior_answer to true ONLY if this utterance adds or "
                "changes a detail/constraint of that same topic (e.g. 'actually the trip was "
                "international'). Set it to false for a new, unrelated question."
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
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 Gemini parse failure on final chunk ({e}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 parse failure, defaulting to WAIT: {e}",
            )
        except RuntimeError:
            # Auth/config errors (bad key) are NOT transient: surface them
            # loudly instead of silently degrading forever.
            raise
        except Exception as e:
            # Transient LLM outage (503 after retries, timeout, network).
            # Never crash the turn. If the utterance is finished, we still
            # need an answer, so fall back to the offline heuristic
            # (RETRIEVE_FINAL); mid-utterance, WAIT is the safe default.
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 LLM unavailable ({type(e).__name__}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 LLM unavailable ({type(e).__name__}), defaulting to WAIT",
            )


class GroqStage2Confirm:
    """Production Stage 2 confirm powered by Groq (openai/gpt-oss-20b)."""

    def __init__(self, client: Optional[object] = None) -> None:
        if client is None:
            from llm.groq_client import GroqLLMClient
            self._client = GroqLLMClient()
        else:
            self._client = client

    def confirm(
        self, utterance_so_far: str, is_final: bool, prior_context: Optional[str] = None
    ) -> ControllerDecision:
        user_prompt = (
            f"Utterance so far: \"{utterance_so_far}\"\n"
            f"Is this the final chunk of the utterance? {is_final}"
        )
        if prior_context:
            user_prompt += (
                f"\nThe user already received an answer about: {prior_context}\n"
                "Set is_refinement_of_prior_answer to true ONLY if this utterance adds or "
                "changes a detail/constraint of that same topic (e.g. 'actually the trip was "
                "international'). Set it to false for a new, unrelated question."
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
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 Groq parse failure on final chunk ({e}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 parse failure, defaulting to WAIT: {e}",
            )
        except RuntimeError:
            raise
        except Exception as e:
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 Groq LLM unavailable ({type(e).__name__}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 Groq LLM unavailable ({type(e).__name__}), defaulting to WAIT",
            )


class OpenAIStage2Confirm:
    """Production Stage 2 confirm powered by OpenAI (gpt-4o-mini)."""

    def __init__(self, client: Optional[object] = None) -> None:
        if client is None:
            from llm.openai_client import OpenAILLMClient
            self._client = OpenAILLMClient()
        else:
            self._client = client

    def confirm(
        self, utterance_so_far: str, is_final: bool, prior_context: Optional[str] = None
    ) -> ControllerDecision:
        user_prompt = (
            f"Utterance so far: \"{utterance_so_far}\"\n"
            f"Is this the final chunk of the utterance? {is_final}"
        )
        if prior_context:
            user_prompt += (
                f"\nThe user already received an answer about: {prior_context}\n"
                "Set is_refinement_of_prior_answer to true ONLY if this utterance adds or "
                "changes a detail/constraint of that same topic (e.g. 'actually the trip was "
                "international'). Set it to false for a new, unrelated question."
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
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 OpenAI parse failure on final chunk ({e}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 parse failure, defaulting to WAIT: {e}",
            )
        except RuntimeError:
            raise
        except Exception as e:
            if is_final:
                fallback = HeuristicStage2Confirm().confirm(utterance_so_far, is_final, prior_context)
                fallback.reason = f"stage2 OpenAI LLM unavailable ({type(e).__name__}); heuristic fallback"
                return fallback
            return ControllerDecision(
                decision=Decision.WAIT, confidence=0.0,
                reason=f"stage2 OpenAI LLM unavailable ({type(e).__name__}), defaulting to WAIT",
            )


class HeuristicStage2Confirm:
    """OFFLINE DEV FALLBACK. See module docstring."""

    # Step 7 offline stand-in for the LLM's refinement judgement. Only consulted
    # when the session already has an answer (prior_context is set).
    _REFINEMENT_CUE_RE = re.compile(
        r"\b(actually|wait|i meant|correction|instead|in that case|what if|one more thing|except|unless|turns out)\b",
        re.IGNORECASE,
    )

    def confirm(
        self, utterance_so_far: str, is_final: bool, prior_context: Optional[str] = None
    ) -> ControllerDecision:
        word_count = len(utterance_so_far.split())
        is_refinement = bool(prior_context) and bool(self._REFINEMENT_CUE_RE.search(utterance_so_far))
        if is_final:
            return ControllerDecision(
                decision=Decision.RETRIEVE_FINAL,
                confidence=0.9,
                reason="heuristic: is_final flag set",
                is_refinement_of_prior_answer=is_refinement,
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
            is_refinement_of_prior_answer=is_refinement,
        )
