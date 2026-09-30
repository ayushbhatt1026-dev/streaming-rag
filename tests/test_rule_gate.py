"""Run: python -m pytest tests/test_rule_gate.py -v  (or python tests/test_rule_gate.py)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Decision
from controller.rule_gate import RuleGate


def test_too_few_words_waits():
    gate = RuleGate(token_floor=4, stability_window=2)
    assert gate.evaluate("I need") == Decision.WAIT


def test_dangling_connector_waits():
    gate = RuleGate(token_floor=4, stability_window=2)
    assert gate.evaluate("I need to plan a workshop in Pune and") == Decision.WAIT


def test_presentation_only_suppresses():
    gate = RuleGate(token_floor=4, stability_window=2)
    assert gate.evaluate("Can you repeat that in two bullets") == Decision.SUPPRESS


def test_stable_entities_escalates():
    gate = RuleGate(token_floor=4, stability_window=2)
    # Same entities ("Pune", "30") repeated across enough calls to fill the
    # stability window -> should escalate (return None) on the last call.
    # (5 content words each, clears the token_floor=4 check first.)
    r1 = gate.evaluate("I need to plan a customer workshop in Pune for 30 people")
    r2 = gate.evaluate("I need to plan a customer workshop in Pune for 30 people please")
    assert r1 == Decision.WAIT, f"expected WAIT on first call (history not full yet), got {r1}"
    assert r2 is None, f"expected escalation (None), got {r2}"


if __name__ == "__main__":
    test_too_few_words_waits()
    test_dangling_connector_waits()
    test_presentation_only_suppresses()
    test_stable_entities_escalates()
    print("All rule_gate tests passed.")
