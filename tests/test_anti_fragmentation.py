"""Run: python -m pytest tests/test_anti_fragmentation.py -v"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomposer.anti_fragmentation import cap_by_distinctness, merge_near_duplicates
from retrieval.embeddings import TfidfEmbeddingBackend


def test_exact_duplicate_text_merges():
    # The most common real trigger for this guard: a decomposition backend
    # (LLM or heuristic) proposes the literal same sub-query text twice.
    # This must merge under ANY embedding backend (similarity == 1.0).
    eb = TfidfEmbeddingBackend()
    candidates = [
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
        {"text": "cancellation policy for workshop venues in Pune", "intent_type": "policy"},
        {"text": "catering options available on-site", "intent_type": "factual_lookup"},
    ]
    merged = merge_near_duplicates(candidates, eb, threshold=0.9)
    assert len(merged) == 2, f"expected 2 after merge, got {len(merged)}: {merged}"


def test_near_duplicate_paraphrase_is_backend_dependent():
    # HONEST LIMITATION, not a bug: TF-IDF (the offline fallback) has no
    # stemming/lemmatization, so "venues" vs "venue" only scores ~0.70
    # cosine similarity -- below the 0.90 merge threshold. A real semantic
    # embedding (BGE, production) would likely catch this paraphrase and
    # merge it. This test documents the gap rather than hiding it; it
    # asserts the TF-IDF-observed behavior, not the semantically "correct"
    # one. Re-run this same assertion with BGEEmbeddingBackend once you
    # have it installed, and update the expected count if it now merges.
    eb = TfidfEmbeddingBackend()
    candidates = [
        {"text": "cancellation policy for workshop venues", "intent_type": "policy"},
        {"text": "cancellation policy for workshop venue", "intent_type": "policy"},
    ]
    merged = merge_near_duplicates(candidates, eb, threshold=0.9)
    assert len(merged) == 2, (
        "TF-IDF does not merge this paraphrase (expected, documented gap). "
        f"If this now fails with len==1, TF-IDF's behavior changed or you "
        f"swapped in a stronger backend -- update this test accordingly."
    )


def test_cap_keeps_max_k_most_distinct():
    eb = TfidfEmbeddingBackend()
    candidates = [
        {"text": "venue capacity in Pune", "intent_type": "constraint"},
        {"text": "cancellation and refund policy", "intent_type": "policy"},
        {"text": "catering options on-site", "intent_type": "factual_lookup"},
        {"text": "parking availability at the venue", "intent_type": "factual_lookup"},
        {"text": "accessibility features of the venue", "intent_type": "factual_lookup"},
    ]
    capped = cap_by_distinctness(candidates, eb, max_k=4)
    assert len(capped) == 4, f"expected 4, got {len(capped)}"


def test_single_intent_not_padded():
    eb = TfidfEmbeddingBackend()
    candidates = [{"text": "seated capacity of Venue B", "intent_type": "constraint"}]
    merged = merge_near_duplicates(candidates, eb, threshold=0.9)
    capped = cap_by_distinctness(merged, eb, max_k=4)
    assert len(capped) == 1, "single-intent input must not be padded to look like more"


if __name__ == "__main__":
    test_exact_duplicate_text_merges()
    test_near_duplicate_paraphrase_is_backend_dependent()
    test_cap_keeps_max_k_most_distinct()
    test_single_intent_not_padded()
    print("All anti_fragmentation tests passed.")
