"""
M3 — structure-aware chunking.

Rule (per blueprint §2, M3 chunking strategy):
  - Split at headings/sections first, never mid-heading.
  - Target 300-500 tokens per chunk, 15% overlap.
  - Every chunk carries doc_id, section_ref, heading_path.

We approximate "tokens" with whitespace word count. This is a deliberate
simplification: it's good enough to hit the target range and is dependency
-free, but it is NOT the same tokenizer your LLM/embedding model uses. If
you need exact token budgeting for context-window packing later, swap in
your model's real tokenizer (e.g. tiktoken, or the HF tokenizer that ships
with BGE) — the split points here will barely move either way, since we're
already conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from contracts import EvidenceChunk

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
OVERLAP_RATIO = 0.15

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class _Section:
    doc_id: str
    heading_path: str
    section_ref: str
    text: str


def _word_count(text: str) -> int:
    return len(text.split())


def split_into_sections(doc_id: str, raw_text: str) -> List[_Section]:
    """Break a markdown document at heading boundaries. Never splits a
    heading from the text that follows it."""
    matches = list(_HEADING_RE.finditer(raw_text))
    if not matches:
        # No headings at all — treat the whole doc as one section.
        return [_Section(doc_id, doc_id, "Full Document", raw_text.strip())]

    sections: List[_Section] = []
    heading_stack: List[str] = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()

        # Maintain a heading path like "Doc title > Section 2"
        heading_stack = heading_stack[: level - 1] + [title]
        heading_path = " > ".join(heading_stack)

        if body:
            # section_ref: prefer an explicit "Section N" style title if present,
            # else fall back to the raw heading text.
            section_ref = title
            sections.append(_Section(doc_id, heading_path, section_ref, body))

    return sections


def _windowed_chunks(words: List[str]) -> List[List[str]]:
    """Slide a 300-500 word window over `words` with 15% overlap."""
    if len(words) <= TARGET_MAX_TOKENS:
        return [words]

    step = int(TARGET_MAX_TOKENS * (1 - OVERLAP_RATIO))
    windows = []
    i = 0
    while i < len(words):
        window = words[i : i + TARGET_MAX_TOKENS]
        if len(window) < TARGET_MIN_TOKENS and windows:
            # Too small a tail — merge into the previous window instead of
            # emitting a tiny orphan chunk.
            windows[-1] = windows[-1] + window
            break
        windows.append(window)
        if i + TARGET_MAX_TOKENS >= len(words):
            break
        i += step
    return windows


def chunk_document(doc_id: str, raw_text: str) -> List[EvidenceChunk]:
    """Full pipeline: heading split -> token-window split -> EvidenceChunk
    objects with citation metadata attached."""
    chunks: List[EvidenceChunk] = []
    sections = split_into_sections(doc_id, raw_text)

    for sec_idx, section in enumerate(sections):
        words = section.text.split()
        for win_idx, window in enumerate(_windowed_chunks(words)):
            chunk_id = f"{doc_id}::s{sec_idx}::c{win_idx}"
            text = " ".join(window)
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_ref=section.section_ref,
                    heading_path=section.heading_path,
                    text=text,
                )
            )
    return chunks


def load_and_chunk_corpus(corpus_dir: str) -> List[EvidenceChunk]:
    """Read every .md file in corpus_dir and chunk it. doc_id is derived
    from the filename (e.g. doc_01_venue_capacity.md -> Doc_01)."""
    all_chunks: List[EvidenceChunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        stem = path.stem  # e.g. doc_01_venue_capacity
        doc_num = stem.split("_")[1] if "_" in stem else stem
        doc_id = f"Doc_{doc_num}"
        raw_text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_document(doc_id, raw_text))
    return all_chunks


if __name__ == "__main__":
    import sys

    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    chunks = load_and_chunk_corpus(corpus_dir)
    print(f"Loaded {len(chunks)} chunks from {corpus_dir}")
    for c in chunks[:5]:
        print(f"  {c.chunk_id:30s} ({_word_count(c.text):3d} words)  {c.heading_path}")
