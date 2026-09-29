# Streaming Live RAG — M3 Retrieval Slice (Step 1)

This replaces `MockRetriever` with a real local search engine over a
5-document mini-corpus (`corpus/`), covering the exact domain the guide's
own worked examples use (venue booking, cancellation, catering,
reimbursement), so your Phase 2 labelled test sets (guide §7.1) can reuse
these scenarios directly.

## Run it right now (offline, no installs beyond scikit-learn)

```bash
python demo_retrieval.py
```

This uses `TfidfEmbeddingBackend` + `InMemoryVectorStore` — offline dev
fallbacks, clearly labelled as such in `retrieval/embeddings.py` and
`retrieval/vector_store.py`. They exist only because this sandbox has no
internet access to download BGE's model weights. They prove the plumbing
(chunking, dense+sparse search, RRF fusion, dedupe, corpus isolation)
actually works — the classical TF-IDF vectors will miss synonyms/paraphrases
that BGE would catch, so don't use these for reporting real G3/G4 numbers.

## Switch to the production stack (decision log)

```bash
pip install -r requirements.txt
```

Then in `demo_retrieval.py` (or wherever you construct the `Retriever`),
change two lines:

```python
from retrieval.embeddings import BGEEmbeddingBackend   # was TfidfEmbeddingBackend
from retrieval.vector_store import QdrantVectorStore    # was InMemoryVectorStore

retriever = Retriever(
    embedding_backend=BGEEmbeddingBackend(),
    vector_store=QdrantVectorStore(),
)
```

First run will download `BAAI/bge-small-en-v1.5` (~130MB) — needs internet
once, then it's cached locally and needs no further network access.
`QdrantVectorStore` runs embedded/local by default (writes to
`./qdrant_data/`) — no Docker or server required unless you want the
Qdrant dashboard UI.

## Prove corpus isolation yourself

```bash
python -m pytest tests/test_corpus_isolation.py -v
```

(or, if pytest isn't installed, run the same assertions manually — see the
test file, it's ~15 lines and doesn't need pytest's machinery to be
meaningful.)

This monkeypatches `socket.socket` to raise on any network attempt, then
runs a full index + search cycle. It passing means the retriever's answer
came from nowhere but `corpus/`.

## What's real vs. what's a placeholder

| Piece | Status |
|---|---|
| Structure-aware chunking (300-500 tokens, 15% overlap, heading-aware) | Real, fully implemented |
| BM25 sparse search | Real, pure-Python, zero dependencies, works identically in prod |
| RRF fusion | Real, standard k=60 formula |
| Dedupe (exact + near-duplicate via embedding similarity) | Real |
| Dense search (Qdrant/BGE) | Real code path, needs `pip install` + one-time internet to actually run |
| Dense search (TF-IDF/in-memory) | Real, offline, but NOT semantic — dev/demo only |
| Reranker | `NoOpReranker` stub (keeps RRF order) — swap in a cross-encoder before Phase 3 benchmarking |
| Query cache | Real (per-session, in `Retriever._search_one`) |

## Next: Step 2

Decision log fields for Vector DB, Embedding model, Chunk size/overlap are
now implemented as described above. Main LLM and Controller/Decomposer LLM
choices don't touch this module — they'll matter starting with M1 (the
Controller) and M5 (the Synthesizer).
