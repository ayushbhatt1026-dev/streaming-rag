"""
M3 — embedding backends.

Decision log choice: BGE-small-en-v1.5 via sentence-transformers.

Both backends below implement the same tiny interface (`encode`), so nothing
else in the retrieval pipeline needs to know which one is active. Swap by
changing one line where the retriever is constructed (see retriever.py).

IMPORTANT — read this before you run anything on your own machine:
  `BGEEmbeddingBackend` is the one you actually ship. It requires
  `pip install sentence-transformers` and an internet connection the first
  time it runs (to download the model weights from Hugging Face, ~130MB).
  Once downloaded, it's cached locally and needs no further internet access
  — this satisfies "local = zero latency variance, no rate limits" from the
  tooling table.

  `TfidfEmbeddingBackend` exists ONLY because this sandbox has no internet
  access to download model weights. It is a classical bag-of-words vector,
  not a semantic embedding — it will miss synonyms and paraphrases that BGE
  would catch. Use it to sanity-check the pipeline's plumbing (chunking,
  fusion, dedupe) offline, then switch to BGEEmbeddingBackend for anything
  you intend to actually measure G3/G4 against.
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) L2-normalized float32 array."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BGEEmbeddingBackend:
    """Production backend. Requires: pip install sentence-transformers"""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run:\n"
                "  pip install sentence-transformers\n"
                "This also needs internet access the first time, to download "
                f"{self.MODEL_NAME} from Hugging Face."
            ) from e
        self._model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: List[str]) -> np.ndarray:
        # BGE models recommend a query instruction prefix for asymmetric
        # search; for simplicity here we encode chunks and queries the same
        # way. For a quality bump, prefix queries with:
        #   "Represent this sentence for searching relevant passages: "
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class TfidfEmbeddingBackend:
    """OFFLINE DEV FALLBACK ONLY. Bag-of-words, not semantic. See module
    docstring. Requires: scikit-learn (already in most environments)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus_texts: List[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit-on-the-fly fallback so ad-hoc queries don't crash; quality
            # is best when fit() was called on the full corpus first.
            self._vectorizer.fit(texts)
            self._fitted = True
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(mat)
