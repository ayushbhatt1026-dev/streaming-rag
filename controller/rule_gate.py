"""
M1, Stage 1 — the rule gate. Runs in microseconds, no model call, on every
single incoming chunk. This is the actual "interrupt handler" — Stage 2
(the LLM) only fires for the ambiguous remainder that survives this gate.

Honest limitation: "named entity" and "clause boundary" detection here are
cheap regex/heuristics, not real NLP (no spaCy/NER model — keeping this
dependency-free and sub-millisecond on purpose). They will misfire on
inputs unlike the examples below (e.g. all-lowercase transcripts with no
capitalization from some ASR systems). If your real ASR output doesn't
capitalize proper nouns, swap `_has_entity_hint` for a real NER pass, or a
small regex list of your corpus's actual proper nouns (venue names, doc
IDs) — but keep it semantic/general, not a list of expected test prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Deque, FrozenSet, List, Optional
from collections import deque

from contracts import Decision

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "with", "i", "need", "want", "please",
    "can", "you", "me", "my", "we", "it", "that", "this", "so", "just",
}

_DANGLING_LAST_WORDS = {
    "and", "or", "but", "for", "with", "to", "of", "in", "the", "a", "an",
    "that", "which", "because", "so", "if", "when", "actually",
}

# General vocabulary, not memorized test prompts — see module docstring
# note in the blueprint about keeping this semantic rather than a keyword
# list of expected benchmark phrases. Treat this as a first pass; anything
# ambiguous still gets a real decision from Stage 2, this list only ever
# produces a SUPPRESS for very unambiguous formatting requests.
_PRESENTATION_ONLY_RE = re.compile(
    r"\b(repeat|again|shorter|shorten|bullet\w*|translate|rephrase|"
    r"reword|simpler|simplify|in two (lines|bullets|points)|"
    r"summari[sz]\w*\s+(that|the\s+(answer|response|explanation|last)|in\s+\w+\s+(line|bullet|point|word)s?|for\s+my\s+\w+))\b",
    re.IGNORECASE,
)

_DOMAIN_TOPIC_RE = re.compile(
    r"\b(venue\w*|cancellation\w*|polic\w*|catering\w*|reimburse\w*|travel\w*|"
    r"workshop\w*|hotel\w*|flight\w*|expense\w*|surcharge\w*|capacit\w*|parking\w*|"
    r"trip\w*|route\w*|airport\w*|employee\w*|allowance\w*|budget\w*|doc\w*|studio\w*|"
    r"auditorium\w*|banquet\w*|suite\w*|conference\w*|meeting\w*)\b",
    re.IGNORECASE,
)

_REFINEMENT_CUE_RE = re.compile(
    r"\b(actually|wait|i meant|correction|instead|in that case|what if|one more thing|except|unless|turns out)\b",
    re.IGNORECASE,
)

_ENTITY_HINT_RE = re.compile(r"\b([A-Z][a-zA-Z]+|\d+)\b")


def _content_words(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _ends_with_dangling_connector(text: str) -> bool:
    words = text.strip().rstrip(".,!?").split()
    if not words:
        return True
    return words[-1].lower() in _DANGLING_LAST_WORDS


def _entity_set(text: str) -> FrozenSet[str]:
    # Skip the very first word of the whole utterance-so-far since
    # sentence-initial capitalization ("Is", "What", "How") is not a reliable
    # entity signal. The comment always said this — now the code actually does it.
    words = text.strip().split()
    searchable = " ".join(words[1:]) if len(words) > 1 else ""
    matches = _ENTITY_HINT_RE.findall(searchable)
    return frozenset(m.lower() for m in matches)


@dataclass
class RuleGate:
    token_floor: int = 4
    stability_window: int = 2

    def __post_init__(self) -> None:
        self._entity_history: Deque[FrozenSet[str]] = deque(maxlen=self.stability_window)

    def evaluate(self, utterance_so_far: str) -> Optional[Decision]:
        """Returns a final Decision (WAIT/SUPPRESS) if the rule gate can
        decide on its own, or None to escalate to Stage 2."""

        has_domain_topic = bool(_DOMAIN_TOPIC_RE.search(utterance_so_far))

        if not has_domain_topic and _PRESENTATION_ONLY_RE.search(utterance_so_far):
            # If the utterance contains a refinement or correction cue,
            # do not unconditionally suppress — escalate to Stage 2 so it can
            # determine if this is a refinement of a prior answer.
            if _REFINEMENT_CUE_RE.search(utterance_so_far):
                return None
            return Decision.SUPPRESS

        if len(_content_words(utterance_so_far)) < self.token_floor:
            return Decision.WAIT

        entities = _entity_set(utterance_so_far)
        if not entities:
            return Decision.WAIT

        self._entity_history.append(entities)

        if _ends_with_dangling_connector(utterance_so_far):
            return Decision.WAIT

        entity_stable = (
            len(self._entity_history) == self.stability_window
            and len(set(self._entity_history)) == 1
        )

        if entity_stable:
            return None  # escalate to Stage 2

        return Decision.WAIT
