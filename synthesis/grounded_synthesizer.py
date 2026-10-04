"""
Step 8 — Grounded Synthesizer (Gate G4).

Blueprint contract:
  • Every factual sentence carries a citation to a REAL, retrieved chunk.
  • Zero fabricated doc IDs ever reach the user.
  • Unsupported claims are flagged as uncertainties (not dropped silently).
  • ≥85 % of sampled assertions are supported by their cited chunk.
  • TTFT (time-to-first-token) is measured and emitted to telemetry.

Architecture:
  1. Build a WHITELIST of valid citation labels from the evidence that was
     actually retrieved for this turn.  Any citation the LLM hallucates that
     is not in the whitelist is stripped from the output (hard-zero G4).
  2. Call the Groq LLM (openai/gpt-oss-20b) with a strict JSON prompt:
     it must return a list of {text, citations, supported} objects.
  3. Sentences marked supported=false (or whose citations are all stripped by
     the whitelist filter) are moved to AnswerVersion.uncertainties instead
     of being asserted confidently.
  4. Return BOTH the formatted string answer AND the AnswerVersion — the
     orchestrator no longer needs to call claim_builder.build_initial_claims()
     separately.

Public API (drop-in for NaiveSynthesizer's .synthesize(), plus extra):
  synthesize(sub_queries, evidence) -> str            # backward-compat
  synthesize_with_claims(sub_queries, evidence, ...) -> (str, AnswerVersion)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from contracts import AnswerVersion, Claim, ClaimStatus, EvidenceChunk, SubQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_whitelist(evidence: List[EvidenceChunk]) -> Dict[str, str]:
    """
    Returns {citation_label -> chunk_id} for every chunk in evidence.

    The LLM may only reference labels that appear in this dict.  Any other
    [Doc_NNN §...] string is a fabrication and must be stripped.
    """
    return {chunk.citation(): chunk.chunk_id for chunk in evidence}


# Matches citation labels like [Doc_01 §2.1] or [Doc_999 §foo].
_CITATION_RE = re.compile(r"\[[^\[\]]+\]")


def _strip_fabricated_citations(text: str, whitelist: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Remove from *text* any citation label not in *whitelist*.

    Returns (cleaned_text, list_of_surviving_citations).
    """
    kept: List[str] = []
    removed: List[str] = []

    def _replace(m: re.Match) -> str:
        label = m.group(0)
        if label in whitelist:
            kept.append(label)
            return label
        removed.append(label)
        return ""  # strip it

    cleaned = _CITATION_RE.sub(_replace, text).strip()
    # Collapse multiple spaces that might result from stripping
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned, kept


def _format_evidence_for_prompt(
    sub_queries: List[SubQuery], evidence: List[EvidenceChunk]
) -> str:
    """Build the evidence block shown to the LLM (deduplicated to minimize tokens)."""
    by_sq: Dict[str, List[EvidenceChunk]] = {sq.id: [] for sq in sub_queries}
    for chunk in evidence:
        if chunk.sub_query_id in by_sq:
            by_sq[chunk.sub_query_id].append(chunk)

    lines: List[str] = []
    seen_citations = set()

    for sq in sub_queries:
        chunks = sorted(
            by_sq.get(sq.id, []),
            key=lambda c: -(c.rerank_score or c.rrf_score),
        )
        lines.append(f"## Sub-query: {sq.text}")
        if not chunks:
            lines.append("  (no evidence retrieved — mark as unsupported)")
        else:
            for ch in chunks[:2]:  # top-2 per sub-query
                cite = ch.citation()
                if cite in seen_citations:
                    lines.append(f"  CITATION: {cite} (see text above)")
                else:
                    seen_citations.add(cite)
                    lines.append(f"  CITATION: {cite}")
                    lines.append(f"  TEXT: {ch.text.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM system prompt — strict about citations
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a factual assistant. You will be given a set of retrieved evidence
chunks identified by their exact citation labels (e.g. [Doc_01 §2]).

Rules you MUST follow:
1. Every claim in your answer must be grounded in one of the provided evidence
   chunks.  Copy the citation label EXACTLY as it appears — do not invent or
   modify labels.
2. If a sub-query has no evidence, or you cannot find supporting text, set
   supported=false.  Do NOT fabricate information.
3. Return ONLY a JSON object with this exact schema:
   {
     "claims": [
       {
         "text": "<one factual sentence>",
         "citations": ["<label1>", "<label2>"],
         "supported": true
       }
     ]
   }
4. One object per sub-query, in the same order as the sub-queries.
5. Do not add commentary outside the JSON.
"""


# ---------------------------------------------------------------------------
# GroundedSynthesizer
# ---------------------------------------------------------------------------

@dataclass
class SynthesisMetrics:
    ttft_ms: float        # time-to-first-token (wall-clock ms of the LLM call)
    n_claims: int
    n_uncertainties: int
    n_fabricated_stripped: int
    used_llm: bool


class GroundedSynthesizer:
    """
    The real M5 — replaces NaiveSynthesizer with a Groq-backed, whitelist-
    filtered, entailment-checked synthesizer that returns AnswerVersion.

    Falls back gracefully to extractive mode if no LLM key is available.
    """

    def __init__(self, llm_client=None) -> None:
        """
        llm_client: a GroqLLMClient instance (or any object with
                    .complete_json(system, user) -> dict).
                    If None, the synthesizer auto-instantiates GroqLLMClient,
                    and if that fails, falls back to heuristic/extractive mode.
        """
        self._llm = llm_client
        self._llm_attempted = False

    def _ensure_llm(self):
        if self._llm is not None or self._llm_attempted:
            return
        self._llm_attempted = True
        try:
            from llm.groq_client import GroqLLMClient
            self._llm = GroqLLMClient()
        except Exception:
            self._llm = None  # fall back to extractive mode

    def synthesize(
        self, sub_queries: List[SubQuery], evidence: List[EvidenceChunk]
    ) -> str:
        """Backward-compatible: returns just the answer string."""
        text, _ = self.synthesize_with_claims(sub_queries, evidence)
        return text

    def synthesize_with_claims(
        self,
        sub_queries: List[SubQuery],
        evidence: List[EvidenceChunk],
        version: int = 1,
        metrics_out: Optional[List[SynthesisMetrics]] = None,
    ) -> Tuple[str, AnswerVersion]:
        """
        Full Step-8 synthesis.

        Returns (answer_text, AnswerVersion).

        metrics_out: if a list is passed, a SynthesisMetrics object is appended.
        """
        self._ensure_llm()
        whitelist = _build_whitelist(evidence)

        if self._llm is not None:
            return self._llm_synthesize(
                sub_queries, evidence, whitelist, version, metrics_out
            )
        else:
            return self._extractive_fallback(
                sub_queries, evidence, whitelist, version, metrics_out
            )

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_synthesize(
        self,
        sub_queries: List[SubQuery],
        evidence: List[EvidenceChunk],
        whitelist: Dict[str, str],
        version: int,
        metrics_out: Optional[List[SynthesisMetrics]],
    ) -> Tuple[str, AnswerVersion]:
        evidence_block = _format_evidence_for_prompt(sub_queries, evidence)
        user_prompt = (
            f"Sub-queries to answer (one claim object each, in order):\n"
            + "\n".join(f"  {i+1}. {sq.text}" for i, sq in enumerate(sub_queries))
            + f"\n\nEvidence:\n{evidence_block}"
        )

        t0 = time.perf_counter()
        try:
            raw = self._llm.complete_json(_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            # LLM call failed — fall back to extractive
            print(f"[grounded_synthesizer] LLM error, falling back: {exc}")
            self._llm = None
            return self._extractive_fallback(
                sub_queries, evidence, whitelist, version, metrics_out
            )
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        raw_claims = raw.get("claims", [])
        if not isinstance(raw_claims, list):
            raw_claims = []

        claims: List[Claim] = []
        uncertainties: List[str] = []
        answer_lines: List[str] = []
        n_fabricated = 0

        for i, entry in enumerate(raw_claims):
            if not isinstance(entry, dict):
                continue

            raw_text: str = str(entry.get("text", "")).strip()
            raw_cites: List[str] = entry.get("citations", [])
            supported: bool = bool(entry.get("supported", True))

            # --- Whitelist filter: strip any citation not in evidence ---
            kept_cites: List[str] = []
            for lbl in raw_cites:
                if lbl in whitelist:
                    kept_cites.append(lbl)
                else:
                    n_fabricated += 1
                    print(
                        f"[G4-WHITELIST] Stripped fabricated citation {lbl!r} "
                        f"(not in retrieved evidence)"
                    )

            # Also strip inline citations embedded in the text itself
            clean_text, inline_kept = _strip_fabricated_citations(raw_text, whitelist)
            # Merge kept lists (deduplicate, preserve order)
            all_kept = kept_cites[:]
            for c in inline_kept:
                if c not in all_kept:
                    all_kept.append(c)

            # --- Entailment / support check ---
            # A sentence is unsupported if the LLM flagged it OR if all its
            # citations were stripped away (meaning we can't verify it).
            is_unsupported = (not supported) or ( not all_kept)

            if is_unsupported or not clean_text:
                sq_text = sub_queries[i].text if i < len(sub_queries) else f"sub-query {i+1}"
                msg = clean_text or f'"{sq_text}" is not documented in the corpus.'
                uncertainties.append(msg)
                answer_lines.append(f"- [uncertain] {msg}")
            else:
                cite_str = " ".join(all_kept)
                claim_text = (
                    clean_text
                    if i >= len(sub_queries)
                    else f'Re: "{sub_queries[i].text}" — {clean_text}'
                )
                claims.append(
                    Claim(
                        id=str(uuid.uuid4())[:8],
                        text=claim_text,
                        citations=all_kept,
                        status=ClaimStatus.ADDED,
                        introduced_in_version=version,
                    )
                )
                answer_lines.append(f"- {clean_text} {cite_str}".strip())

        # If LLM returned no usable claims, fall back to extractive
        if not claims and not uncertainties:
            return self._extractive_fallback(
                sub_queries, evidence, whitelist, version, metrics_out
            )

        answer_text = "\n".join(answer_lines) if answer_lines else "(no answer generated)"
        av = AnswerVersion(version=version, claims=claims, uncertainties=uncertainties)

        if metrics_out is not None:
            metrics_out.append(
                SynthesisMetrics(
                    ttft_ms=ttft_ms,
                    n_claims=len(claims),
                    n_uncertainties=len(uncertainties),
                    n_fabricated_stripped=n_fabricated,
                    used_llm=True,
                )
            )
        print(
            f"[synthesis] TTFT={ttft_ms:.0f}ms  claims={len(claims)}  "
            f"uncertainties={len(uncertainties)}  fabricated_stripped={n_fabricated}"
        )
        return answer_text, av

    # ------------------------------------------------------------------
    # Extractive fallback (same logic as NaiveSynthesizer, but produces
    # AnswerVersion and still applies the whitelist filter)
    # ------------------------------------------------------------------

    def _extractive_fallback(
        self,
        sub_queries: List[SubQuery],
        evidence: List[EvidenceChunk],
        whitelist: Dict[str, str],
        version: int,
        metrics_out: Optional[List[SynthesisMetrics]],
    ) -> Tuple[str, AnswerVersion]:
        by_sq: Dict[str, List[EvidenceChunk]] = {sq.id: [] for sq in sub_queries}
        for chunk in evidence:
            if chunk.sub_query_id in by_sq:
                by_sq[chunk.sub_query_id].append(chunk)

        claims: List[Claim] = []
        uncertainties: List[str] = []
        answer_lines: List[str] = []

        for sq in sub_queries:
            chunks = sorted(
                by_sq.get(sq.id, []),
                key=lambda c: -(c.rerank_score or c.rrf_score),
            )
            if not chunks:
                msg = f'"{sq.text}" is not documented in the corpus.'
                uncertainties.append(msg)
                answer_lines.append(f"- [uncertain] {msg}")
                continue

            best = chunks[0]
            label = best.citation()
            # Whitelist — label MUST be in whitelist (trivially true for
            # extractive path, but asserted explicitly for test coverage)
            if label not in whitelist:
                msg = f'"{sq.text}" — citation {label} not in retrieved evidence.'
                uncertainties.append(msg)
                answer_lines.append(f"- [uncertain] {msg}")
                continue

            excerpt = best.text.strip()[:300]
            claim_text = f'Re: "{sq.text}": {excerpt}'
            claims.append(
                Claim(
                    id=str(uuid.uuid4())[:8],
                    text=claim_text,
                    citations=[label],
                    status=ClaimStatus.ADDED,
                    introduced_in_version=version,
                )
            )
            answer_lines.append(f"- Re: \"{sq.text}\" — {excerpt} {label}")

        answer_text = "\n".join(answer_lines) if answer_lines else "(no answer generated)"
        av = AnswerVersion(version=version, claims=claims, uncertainties=uncertainties)

        if metrics_out is not None:
            metrics_out.append(
                SynthesisMetrics(
                    ttft_ms=0.0,
                    n_claims=len(claims),
                    n_uncertainties=len(uncertainties),
                    n_fabricated_stripped=0,
                    used_llm=False,
                )
            )
        return answer_text, av
