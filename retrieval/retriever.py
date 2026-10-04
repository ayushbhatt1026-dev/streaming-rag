"""
M3 — the real Retriever.

This is what WS-A ships to unblock WS-B/WS-C in Phase 1. It used to be a
MockRetriever returning fixed chunks; this class has the same shape
(`search(sub_queries) -> List[EvidenceChunk]`) so nothing downstream (the
Decomposer's caller, the Synthesizer) needs to change when you swap it in.

Corpus isolation, enforced by construction, not just by policy:
  - This module never imports `requests`, `urllib`, `httpx`, or any network
    client. Its only data sources are the local vector store and the local
    BM25 index, both built from `corpus/` at index time.
  - `search()` takes only text and returns only chunks that trace back to a
    doc_id/section_ref pulled from the local corpus. There is no code path
    by which an external fact could enter an EvidenceChunk.
  - See tests/test_corpus_isolation.py for an automated check of this.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from contracts import EvidenceChunk, SubQuery
from retrieval.chunking import load_and_chunk_corpus
from retrieval.embeddings import EmbeddingBackend
from retrieval.fusion import NoOpReranker, Reranker, dedupe_near_duplicates, reciprocal_rank_fusion
from retrieval.sparse_bm25 import BM25Index
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        top_k_final: int = 8,
    ) -> None:
        self._embed = embedding_backend
        self._vector_store = vector_store
        self._bm25 = BM25Index()
        self._reranker: Reranker = reranker or NoOpReranker()
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_final = top_k_final

        self._chunks_by_id: Dict[str, EvidenceChunk] = {}
        self._chunk_vectors: Dict[str, np.ndarray] = {}
        self._query_cache: Dict[str, List[EvidenceChunk]] = {}
        self.query_log: List[dict] = []

    def clear_query_log(self) -> None:
        self.query_log.clear()

    def index_corpus(self, corpus_dir: str = "corpus") -> int:
        """Chunk every document, embed once, and build both indexes.
        Call this once at startup — not per query."""
        chunks = load_and_chunk_corpus(corpus_dir)
        if not chunks:
            raise ValueError(f"No .md documents found under '{corpus_dir}'")

        texts = [c.text for c in chunks]
        # sentence-transformers-style backends don't need fit(); the
        # offline TF-IDF fallback does — call it if present, harmlessly
        # ignored otherwise.
        if hasattr(self._embed, "fit"):
            self._embed.fit(texts)  # type: ignore[attr-defined]
        vectors = self._embed.encode(texts)

        ids = [c.chunk_id for c in chunks]
        self._vector_store.upsert(ids, vectors)
        self._bm25.index(ids, texts)

        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._chunk_vectors = {cid: vec for cid, vec in zip(ids, vectors)}
        return len(chunks)

    def _search_one(self, sub_query: SubQuery) -> List[EvidenceChunk]:
        cache_key = sub_query.text.strip().lower()
        if cache_key in self._query_cache:
            # Cache hit: the retrieval work (dense+sparse+RRF+rerank) is
            # genuinely reusable when the query text repeats, but
            # sub_query_id is per-call identity, not part of what makes two
            # queries "the same." Re-stamp it on every hit via
            # dataclasses.replace (a shallow copy) so we never mutate the
            # cached original — without this, a second sub-query that
            # happens to share text with an earlier one (very common here,
            # since the controller re-decomposes as the utterance grows)
            # silently loses its evidence downstream: the synthesizer looks
            # up chunks by the CURRENT sub_query.id, but the cached chunks
            # would still carry the FIRST call's id.
            hits = [replace(c, sub_query_id=sub_query.id) for c in self._query_cache[cache_key]]
            self.query_log.append({
                "sub_query": sub_query.text,
                "sub_query_id": sub_query.id,
                "cache_hit": True,
                "n_results": len(hits),
            })
            return hits

        q_vec = self._embed.encode([sub_query.text])[0]
        dense_hits = self._vector_store.search(q_vec, self.top_k_dense)
        sparse_hits = self._bm25.search(sub_query.text, self.top_k_sparse)

        fused_scores = reciprocal_rank_fusion([dense_hits, sparse_hits])
        dense_score_by_id = dict(dense_hits)
        sparse_score_by_id = dict(sparse_hits)

        result_chunks: List[EvidenceChunk] = []
        for chunk_id, rrf_score in fused_scores.items():
            base = self._chunks_by_id[chunk_id]
            result_chunks.append(
                EvidenceChunk(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    section_ref=base.section_ref,
                    heading_path=base.heading_path,
                    text=base.text,
                    dense_score=dense_score_by_id.get(chunk_id, 0.0),
                    sparse_score=sparse_score_by_id.get(chunk_id, 0.0),
                    rrf_score=rrf_score,
                    sub_query_id=sub_query.id,
                )
            )

        result_chunks = dedupe_near_duplicates(result_chunks, self._chunk_vectors)
        result_chunks = self._reranker.rerank(sub_query.text, result_chunks)
        result_chunks = result_chunks[: self.top_k_final]

        self._query_cache[cache_key] = result_chunks
        self.query_log.append({
            "sub_query": sub_query.text,
            "sub_query_id": sub_query.id,
            "cache_hit": False,
            "n_results": len(result_chunks),
        })
        return result_chunks

    def search(self, sub_queries: List[SubQuery]) -> List[EvidenceChunk]:
        """Fan out across sub-queries, union results, dedupe again across
        the union, cap the global result set. This is what M2 (the
        Decomposer) calls after splitting a compound utterance."""
        all_chunks: List[EvidenceChunk] = []
        for sq in sub_queries:
            all_chunks.extend(self._search_one(sq))

        all_chunks = dedupe_near_duplicates(all_chunks, self._chunk_vectors)
        all_chunks.sort(key=lambda c: -(c.rerank_score or c.rrf_score))
        return all_chunks[: max(self.top_k_final, len(sub_queries) * 3)]
