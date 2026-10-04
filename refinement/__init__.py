"""
M4 — Refinement Engine package.
Provides claim-graph management, touch classification, contradiction checking,
and delta refinement across conversational turns.
"""

from refinement.claim_builder import build_initial_claims
from refinement.contradiction_checker import LLMContradictionChecker
from refinement.refinement_engine import RefinementEngine
from refinement.session_store import SessionStore
from refinement.touch_classifier import find_touched_claims

__all__ = [
    "RefinementEngine",
    "build_initial_claims",
    "find_touched_claims",
    "LLMContradictionChecker",
    "SessionStore",
]
