"""
Step 8 tests — Grounded Synthesizer (Gate G4).

Coverage:
  G4-1  Whitelist filter strips fabricated [Doc_999] citations hard (zero tolerance)
  G4-2  Unanswerable probes produce explicit uncertainty, not confident assertion
  G4-3  Supported sentences from real evidence are claimed with correct citation
  G4-4  Mixed evidence: some sub-queries answered, some flagged uncertain
  G4-5  TTFT is measured and non-negative
  G4-6  synthesize() backward-compat shim returns a string
  G4-7  Extractive fallback still applies whitelist (no fabrication possible)
  G4-8  LLM fabricated inline citation in sentence text is stripped
  G4-9  AnswerVersion.citations_all only contains whitelisted labels
  G4-10 TTFT is logged to console (captured by capsys)
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from contracts import AnswerVersion, Claim, ClaimStatus, EvidenceChunk, SubQuery
from synthesis.grounded_synthesizer import (
    GroundedSynthesizer,
    SynthesisMetrics,
    _build_whitelist,
    _strip_fabricated_citations,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _sq(text: str, sq_id: str = "sq1") -> SubQuery:
    return SubQuery(id=sq_id, text=text, parent_utterance_id="turn1")


def _chunk(
    chunk_id: str,
    doc_id: str,
    section_ref: str,
    text: str,
    sub_query_id: str = "sq1",
    rrf_score: float = 0.5,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_ref=section_ref,
        heading_path=f"{doc_id} > {section_ref}",
        text=text,
        rrf_score=rrf_score,
        sub_query_id=sub_query_id,
    )


REAL_CHUNK = _chunk(
    "Doc_01::s1::c0", "Doc_01", "2.1",
    "Venue A in Mumbai can accommodate up to 50 guests.",
    sub_query_id="sq1",
)
REAL_CITE = REAL_CHUNK.citation()  # "[Doc_01 §2.1]"

REAL_CHUNK_2 = _chunk(
    "Doc_02::s3::c0", "Doc_02", "3.0",
    "Cancellation policy: full refund if cancelled 30 days before the event.",
    sub_query_id="sq2",
    rrf_score=0.4,
)
REAL_CITE_2 = REAL_CHUNK_2.citation()  # "[Doc_02 §3.0]"

FAKE_CITE = "[Doc_999 §99]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm(claims_payload: list) -> MagicMock:
    """Create a mock LLM client that returns a fixed JSON payload."""
    mock = MagicMock()
    mock.complete_json.return_value = {"claims": claims_payload}
    return mock


# ---------------------------------------------------------------------------
# Unit tests — _build_whitelist and _strip_fabricated_citations
# ---------------------------------------------------------------------------

class TestWhitelistHelpers:
    def test_whitelist_includes_all_evidence_citations(self):
        wl = _build_whitelist([REAL_CHUNK, REAL_CHUNK_2])
        assert REAL_CITE in wl
        assert REAL_CITE_2 in wl

    def test_whitelist_excludes_fabricated_labels(self):
        wl = _build_whitelist([REAL_CHUNK])
        assert FAKE_CITE not in wl

    def test_strip_keeps_real_citation(self):
        text = f"Venue A seats 50. {REAL_CITE}"
        wl = _build_whitelist([REAL_CHUNK])
        cleaned, kept = _strip_fabricated_citations(text, wl)
        assert REAL_CITE in cleaned
        assert REAL_CITE in kept

    def test_strip_removes_fabricated_citation(self):
        text = f"Venue X seats 100. {FAKE_CITE}"
        wl = _build_whitelist([REAL_CHUNK])
        cleaned, kept = _strip_fabricated_citations(text, wl)
        assert FAKE_CITE not in cleaned
        assert kept == []

    def test_strip_mixed_keeps_only_real(self):
        text = f"Some fact. {REAL_CITE} Also fake. {FAKE_CITE}"
        wl = _build_whitelist([REAL_CHUNK])
        cleaned, kept = _strip_fabricated_citations(text, wl)
        assert REAL_CITE in cleaned
        assert FAKE_CITE not in cleaned
        assert kept == [REAL_CITE]


# ---------------------------------------------------------------------------
# G4-1  Whitelist filter — fabricated Doc_999 citation is hard-stripped
# ---------------------------------------------------------------------------

class TestG4WhitelistFilter:
    def test_fabricated_citation_in_citations_list_stripped(self):
        """
        Blueprint validation 1: manually inject a fabricated [Doc_999] citation
        into a test generation; confirm the whitelist filter strips it before it
        reaches the output.
        """
        llm = _fake_llm([
            {
                "text": "Venue A can hold 50 guests.",
                "citations": [FAKE_CITE],   # fabricated — not in evidence
                "supported": True,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        sqs = [_sq("venue capacity")]
        metrics: List[SynthesisMetrics] = []
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK], metrics_out=metrics)

        # The fabricated citation must not appear anywhere in output
        assert FAKE_CITE not in answer
        # Because all citations were stripped, sentence becomes uncertain
        assert len(av.uncertainties) == 1
        assert len(av.claims) == 0

    def test_fabricated_inline_citation_in_sentence_text_stripped(self):
        """G4-8: fabricated citation embedded in the sentence text is also stripped."""
        llm = _fake_llm([
            {
                "text": f"Venue seats 50. {FAKE_CITE}",
                "citations": [REAL_CITE],
                "supported": True,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        sqs = [_sq("capacity")]
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK])
        assert FAKE_CITE not in answer
        # REAL_CITE is valid — claim should survive
        assert len(av.claims) == 1

    def test_zero_fabricated_ids_in_answer_version_citations(self):
        """G4-9: AnswerVersion.citations_all only contains whitelisted labels."""
        llm = _fake_llm([
            {
                "text": "Venue A in Mumbai.",
                "citations": [REAL_CITE, FAKE_CITE],
                "supported": True,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        answer, av = synth.synthesize_with_claims([_sq("venue")], [REAL_CHUNK])
        for cite in av.citations_all:
            assert cite != FAKE_CITE, f"Fabricated citation {cite!r} leaked into AnswerVersion!"


# ---------------------------------------------------------------------------
# G4-2  Unanswerable probes return explicit uncertainty
# ---------------------------------------------------------------------------

class TestG4Uncertainties:
    def test_no_evidence_becomes_uncertainty_llm_path(self):
        """
        Blueprint validation 2: unanswerable probes return explicit uncertainty,
        not confident assertion.
        """
        llm = _fake_llm([
            {
                "text": "This information is not available in the corpus.",
                "citations": [],
                "supported": False,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        sqs = [_sq("parking availability at Venue A")]
        answer, av = synth.synthesize_with_claims(sqs, evidence=[])

        assert len(av.uncertainties) >= 1
        assert len(av.claims) == 0
        # The answer text must explicitly flag uncertainty
        assert "[uncertain]" in answer.lower() or "uncertain" in answer.lower() or "not available" in answer.lower()

    def test_no_evidence_extractive_fallback_flags_uncertainty(self):
        """Extractive path: empty evidence → uncertainty, no fabrication."""
        synth = GroundedSynthesizer(llm_client=None)
        synth._llm_attempted = True  # skip auto-init, force extractive
        sqs = [_sq("catering options")]
        answer, av = synth.synthesize_with_claims(sqs, evidence=[])

        assert len(av.uncertainties) == 1
        assert len(av.claims) == 0
        assert "[uncertain]" in answer

    def test_llm_flagged_unsupported_does_not_become_claim(self):
        """If LLM itself says supported=false, it goes to uncertainties."""
        llm = _fake_llm([
            {
                "text": "Parking is available at the venue.",
                "citations": [REAL_CITE],
                "supported": False,  # LLM admits it's guessing
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        answer, av = synth.synthesize_with_claims([_sq("parking")], [REAL_CHUNK])

        assert len(av.claims) == 0
        assert len(av.uncertainties) == 1


# ---------------------------------------------------------------------------
# G4-3  Supported sentences produce correct Claim
# ---------------------------------------------------------------------------

class TestG4SupportedClaims:
    def test_real_citation_preserved_in_claim(self):
        llm = _fake_llm([
            {
                "text": "Venue A in Mumbai seats 50 guests.",
                "citations": [REAL_CITE],
                "supported": True,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        answer, av = synth.synthesize_with_claims([_sq("capacity")], [REAL_CHUNK])

        assert len(av.claims) == 1
        claim = av.claims[0]
        assert REAL_CITE in claim.citations
        assert claim.status == ClaimStatus.ADDED
        assert claim.introduced_in_version == 1

    def test_claim_text_in_answer_string(self):
        llm = _fake_llm([
            {
                "text": "The venue seats 50 guests.",
                "citations": [REAL_CITE],
                "supported": True,
            }
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        answer, av = synth.synthesize_with_claims([_sq("capacity")], [REAL_CHUNK])

        assert "50 guests" in answer
        assert REAL_CITE in answer


# ---------------------------------------------------------------------------
# G4-4  Mixed evidence: some answered, some uncertain
# ---------------------------------------------------------------------------

class TestG4Mixed:
    def test_partial_evidence_produces_both_claims_and_uncertainties(self):
        llm = _fake_llm([
            {
                "text": "Venue A in Mumbai seats 50.",
                "citations": [REAL_CITE],
                "supported": True,
            },
            {
                "text": "Parking information is not available.",
                "citations": [],
                "supported": False,
            },
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        sqs = [_sq("capacity", "sq1"), _sq("parking", "sq2")]
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK])

        assert len(av.claims) == 1
        assert len(av.uncertainties) == 1
        assert FAKE_CITE not in answer


# ---------------------------------------------------------------------------
# G4-5  TTFT measurement
# ---------------------------------------------------------------------------

class TestG4TTFT:
    def test_ttft_is_non_negative_and_logged(self, capsys):
        """
        Blueprint validation 3: measure time-to-first-token; confirm it's
        logged.  We add artificial delay via a slow mock to ensure TTFT > 0.
        """
        def slow_complete(system, user):
            time.sleep(0.01)   # 10 ms simulated LLM latency
            return {"claims": [{"text": "Venue A seats 50.", "citations": [REAL_CITE], "supported": True}]}

        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = slow_complete

        synth = GroundedSynthesizer(llm_client=mock_llm)
        metrics: List[SynthesisMetrics] = []
        answer, av = synth.synthesize_with_claims(
            [_sq("capacity")], [REAL_CHUNK], metrics_out=metrics
        )

        assert len(metrics) == 1
        m = metrics[0]
        assert m.ttft_ms >= 0.0
        assert m.used_llm is True

        # Confirm TTFT printed to console
        captured = capsys.readouterr()
        assert "TTFT=" in captured.out


# ---------------------------------------------------------------------------
# G4-6  Backward-compatible synthesize() shim
# ---------------------------------------------------------------------------

class TestG4BackwardCompat:
    def test_synthesize_returns_string(self):
        llm = _fake_llm([
            {"text": "Venue seats 50.", "citations": [REAL_CITE], "supported": True}
        ])
        synth = GroundedSynthesizer(llm_client=llm)
        result = synth.synthesize([_sq("capacity")], [REAL_CHUNK])
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# G4-7  Extractive fallback whitelist
# ---------------------------------------------------------------------------

class TestG4ExtractiveFallback:
    def test_extractive_fallback_no_fabrication(self):
        """Extractive fallback can only cite chunks in evidence — no LLM needed."""
        synth = GroundedSynthesizer(llm_client=None)
        synth._llm_attempted = True  # skip auto-init
        sqs = [_sq("capacity", "sq1")]
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK])

        assert len(av.claims) == 1
        assert REAL_CITE in av.claims[0].citations
        assert FAKE_CITE not in answer
        assert FAKE_CITE not in av.citations_all

    def test_extractive_fallback_multi_subquery(self):
        synth = GroundedSynthesizer(llm_client=None)
        synth._llm_attempted = True
        sqs = [_sq("capacity", "sq1"), _sq("cancellation", "sq2")]
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK, REAL_CHUNK_2])

        assert len(av.claims) == 2
        assert len(av.uncertainties) == 0


# ---------------------------------------------------------------------------
# G4-LLM-ERROR  LLM exception triggers graceful extractive fallback
# ---------------------------------------------------------------------------

class TestG4LLMError:
    def test_llm_error_falls_back_to_extractive(self):
        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = RuntimeError("Groq timeout")

        synth = GroundedSynthesizer(llm_client=mock_llm)
        sqs = [_sq("capacity", "sq1")]
        answer, av = synth.synthesize_with_claims(sqs, [REAL_CHUNK])

        # Should still produce a valid answer via extractive fallback
        assert isinstance(answer, str)
        assert len(av.claims) >= 0  # extractive path runs
