"""
Unit tests for the evaluation harness (eval/ package).
Verifies that gate scorers, ablation routines, and data structures
run cleanly without exceptions or interface mismatches.
"""

from __future__ import annotations

import pytest

from eval.gates import score_g1
from tests.labelled_sets import (
    COMPOUND,
    EARLY_ELIGIBLE,
    INCOMPLETE,
    PRESENTATION_ONLY,
    heldout,
)


def test_score_g1_deliverables_exist():
    result = score_g1()
    assert result["gate"] == "G1"
    assert result["passed"] is True, f"G1 failed: {result['detail']}"
    assert result["score"] == 1.0


def test_heldout_splits_are_correct_fraction():
    # 30% held-out
    assert len(heldout(EARLY_ELIGIBLE)) == 3
    assert len(heldout(PRESENTATION_ONLY)) == 3
    assert len(heldout(INCOMPLETE)) == 2
    assert len(heldout(COMPOUND)) == 8


def test_heldout_items_do_not_overlap_with_training_prefix():
    items = list(range(10))
    held = heldout(items, 0.3)
    train = items[:-len(held)]
    assert set(held).isdisjoint(set(train))
