"""
M2 — raw sub-query proposal, before the anti-fragmentation guard runs.

Both backends return List[{"text": str, "intent_type": str}], unguarded —
merging/capping happens afterward in decomposer.py, uniformly, regardless
of which backend produced the candidates.
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol

from llm.gemini_client import GeminiLLMClient

_SYSTEM_PROMPT = """You split a user's spoken request into independently answerable,
non-overlapping search queries for a retrieval system.

Rules:
- If the utterance has exactly one intent, return exactly one sub-query — do not pad.
- Each sub-query must be self-contained: resolve pronouns and carry forward topic
  context (e.g. "I'm booking Venue B. What is the cancellation policy and does it have parking?" ->
  "cancellation and refund policy for Venue B", "parking availability for Venue B Skyline Conference Center").
- Speech repairs / conversational corrections: if the user corrects themselves mid-sentence
  (e.g. "Book a venue in Pune... actually Mumbai for 30 people" or "Venue A... wait Venue B"),
  use ONLY the final corrected entity/constraint ("Mumbai", "Venue B") and discard the retracted one ("Pune", "Venue A").
- Tag each with an intent_type: factual_lookup, policy, constraint, or comparison.
- Return at most 4 sub-queries.

Respond ONLY with JSON: {"sub_queries": [{"text": "...", "intent_type": "..."}]}
"""


class DecomposeBackend(Protocol):
    def propose(self, utterance: str) -> List[dict]: ...


class GroqDecomposeBackend:
    """Production Decomposer powered by Groq (openai/gpt-oss-20b)."""

    def __init__(self, client: Optional[object] = None) -> None:
        if client is None:
            from llm.groq_client import GroqLLMClient
            self._client = GroqLLMClient()
        else:
            self._client = client

    def propose(self, utterance: str) -> List[dict]:
        try:
            data = self._client.complete_json(_SYSTEM_PROMPT, f"Utterance: \"{utterance}\"")
            sub_queries = data.get("sub_queries", [])
            if not sub_queries:
                return [{"text": utterance, "intent_type": "factual_lookup"}]
            return sub_queries
        except ValueError:
            return [{"text": utterance, "intent_type": "factual_lookup"}]
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[decomposer] Groq LLM unavailable ({type(e).__name__}); using heuristic fallback")
            return HeuristicDecomposeBackend().propose(utterance)


class OpenAIDecomposeBackend:
    """Production Decomposer powered by OpenAI (gpt-4o-mini)."""

    def __init__(self, client: Optional[object] = None) -> None:
        if client is None:
            from llm.openai_client import OpenAILLMClient
            self._client = OpenAILLMClient()
        else:
            self._client = client

    def propose(self, utterance: str) -> List[dict]:
        try:
            data = self._client.complete_json(_SYSTEM_PROMPT, f"Utterance: \"{utterance}\"")
            sub_queries = data.get("sub_queries", [])
            if not sub_queries:
                return [{"text": utterance, "intent_type": "factual_lookup"}]
            return sub_queries
        except ValueError:
            return [{"text": utterance, "intent_type": "factual_lookup"}]
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[decomposer] OpenAI LLM unavailable ({type(e).__name__}); using heuristic fallback")
            return HeuristicDecomposeBackend().propose(utterance)


class GeminiDecomposeBackend:
    def __init__(self, client: GeminiLLMClient | None = None) -> None:
        self._client = client or GeminiLLMClient()

    def propose(self, utterance: str) -> List[dict]:
        try:
            data = self._client.complete_json(_SYSTEM_PROMPT, f"Utterance: \"{utterance}\"")
            sub_queries = data.get("sub_queries", [])
            if not sub_queries:
                return [{"text": utterance, "intent_type": "factual_lookup"}]
            return sub_queries
        except ValueError:
            # Fail safe: if the model didn't return valid JSON, treat the
            # whole utterance as one intent rather than dropping the turn.
            return [{"text": utterance, "intent_type": "factual_lookup"}]
        except RuntimeError:
            raise  # bad key/config: surface loudly, don't hide it
        except Exception as e:
            # Transient LLM outage/quota after retries: degrade to the
            # offline heuristic so the turn still completes.
            print(f"[decomposer] LLM unavailable ({type(e).__name__}); using heuristic fallback")
            return HeuristicDecomposeBackend().propose(utterance)


class HeuristicDecomposeBackend:
    """OFFLINE DEV FALLBACK. Splits on explicit multi-intent connector
    phrases. No pronoun resolution, no true intent understanding — this is
    a plumbing-test stand-in, not a real decomposer. Swap to
    GeminiDecomposeBackend before measuring real G3 numbers."""

    _SPLIT_RE = re.compile(
        r",?\s+(?:and|as well as|plus|also)\s+", re.IGNORECASE
    )

    _INTENT_KEYWORDS = {
        "policy": ("cancel", "refund", "policy", "reimburs", "approval"),
        "constraint": ("capacity", "venue", "seat", "room", "attendee", "people"),
        "comparison": ("compare", "versus", " vs ", "better", "difference"),
    }

    def propose(self, utterance: str) -> List[dict]:
        parts = [p.strip() for p in self._SPLIT_RE.split(utterance) if p.strip()]
        if len(parts) <= 1:
            return [{"text": utterance.strip(), "intent_type": self._tag(utterance)}]
        return [{"text": p, "intent_type": self._tag(p)} for p in parts]

    def _tag(self, text: str) -> str:
        lowered = text.lower()
        for intent_type, keywords in self._INTENT_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return intent_type
        return "factual_lookup"
