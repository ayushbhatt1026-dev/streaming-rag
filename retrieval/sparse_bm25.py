"""
M3 — sparse (keyword) search via BM25.

The tooling table's recommended default is `rank_bm25` (a tiny pure-Python
package) or Qdrant's native sparse vectors. This is a from-scratch BM25Okapi
implementation with the same math, so:
  (a) it works with zero installed dependencies, including fully offline, and
  (b) you can swap in `rank_bm25.BM25Okapi` later with no interface change —
      just replace the internals of BM25Index, callers don't need to know.

BM25 is what catches exact IDs, codes, and policy names that a dense/semantic
embedding can blur together (e.g. "Doc_04 Section 3" or "Venue B") — this is
exactly why the guide's hybrid-vs-dense-only ablation expects hybrid to win
on those cases.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._doc_ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: List[Counter] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._df: Counter = Counter()  # document frequency per term
        self._idf: Dict[str, float] = {}
        self._n_docs = 0

    def index(self, doc_ids: List[str], texts: List[str]) -> None:
        """Build the index. doc_ids here are chunk_ids, one per text."""
        self._doc_ids = list(doc_ids)
        self._doc_tokens = [tokenize(t) for t in texts]
        self._doc_freqs = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._n_docs = len(texts)
        self._avgdl = sum(self._doc_lens) / max(1, self._n_docs)

        self._df = Counter()
        for freqs in self._doc_freqs:
            for term in freqs.keys():
                self._df[term] += 1

        self._idf = {
            term: math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """Return [(chunk_id, bm25_score), ...] sorted descending, best first."""
        if self._n_docs == 0:
            return []

        q_terms = tokenize(query)
        scores = [0.0] * self._n_docs

        for term in q_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in range(self._n_docs):
                freq = self._doc_freqs[i].get(term, 0)
                if freq == 0:
                    continue
                dl = self._doc_lens[i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(
            zip(self._doc_ids, scores), key=lambda x: x[1], reverse=True
        )
        ranked = [r for r in ranked if r[1] > 0]
        return ranked[:top_k]
