"""
M4 — optional LLM-based contradiction check. Runs ONLY on claims the
embedding touch classifier already flagged — never the whole claim set.
This is the "LLM check on the touched candidates only" the blueprint
specifies, now wired to your current provider (Groq).

Without this attached (contradiction_checker=None in RefinementEngine),
touched claims are conservatively kept as-is — never auto-retracted. That
is the safer default. Only attach this once you've validated it against a
labelled refinement test set (guide §7.1): an overzealous checker
retracting correct claims is a worse failure than never retracting.
"""

from __future__ import annotations

from typing import List, Optional

from contracts import Claim, EvidenceChunk
from llm.groq_client import GroqLLMClient

_SYSTEM_PROMPT = """You check whether a NEW CONSTRAINT contradicts an EXISTING CLAIM.
Narrowing, adding an exception, or adding a new requirement is NOT a contradiction.
Only an outright conflict (the existing claim is now false) counts as a contradiction.
Respond ONLY with JSON: {"contradicts": true or false, "reason": "..."}
"""


class LLMContradictionChecker:
    def __init__(self, client: Optional[GroqLLMClient] = None) -> None:
        self._client = client or GroqLLMClient()

    def check(self, claim: Claim, constraint_text: str, delta_evidence: List[EvidenceChunk]) -> bool:
        evidence_text = " ".join(e.text for e in delta_evidence[:3])
        user_prompt = (
            f'Existing claim: "{claim.text}"\n'
            f'New constraint: "{constraint_text}"\n'
            f'New evidence found: "{evidence_text}"'
        )
        try:
            data = self._client.complete_json(_SYSTEM_PROMPT, user_prompt)
            return bool(data.get("contradicts", False))
        except (ValueError, RuntimeError):
            # Fail safe: if the check itself fails, don't retract. An
            # un-updated KEPT claim is a far safer failure than a wrongly
            # dropped one.
            return False
