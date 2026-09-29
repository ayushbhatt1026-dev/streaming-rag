"""
Runnable, offline-safe demo of the real Retriever replacing MockRetriever.

Uses the OFFLINE DEV FALLBACK backends (TfidfEmbeddingBackend + InMemoryVectorStore)
so this runs with zero internet access and zero extra installs beyond
scikit-learn/numpy. For your actual submission, switch the two lines marked
PRODUCTION below to BGEEmbeddingBackend + QdrantVectorStore.

Run: python demo_retrieval.py
"""

from contracts import SubQuery
from retrieval.embeddings import TfidfEmbeddingBackend  # PRODUCTION: BGEEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import InMemoryVectorStore  # PRODUCTION: QdrantVectorStore


def main() -> None:
    retriever = Retriever(
        embedding_backend=TfidfEmbeddingBackend(),
        vector_store=InMemoryVectorStore(),
        top_k_dense=10,
        top_k_sparse=10,
        top_k_final=5,
    )

    n_chunks = retriever.index_corpus(corpus_dir="corpus")
    print(f"Indexed {n_chunks} chunks from corpus/\n")

    scenarios = [
        # Single-intent, exact-term heavy -> should favor BM25/sparse hits
        [SubQuery(id="q1", text="What is the seated capacity of Venue B Skyline Conference Center?")],
        # Multi-intent, mirrors the guide's own worked example (Example 1)
        [
            SubQuery(id="q2a", text="venue capacity for 30 attendees in Pune", intent_type="constraint"),
            SubQuery(id="q2b", text="cancellation terms and refund policy for workshop venues", intent_type="policy"),
            SubQuery(id="q2c", text="on-site and external catering options", intent_type="factual_lookup"),
        ],
        # Deliberately unanswerable -> should retrieve doc_05 section 4, which
        # explicitly states the gap (this is what your uncertainty probes test)
        [SubQuery(id="q3", text="Is there a weekend booking surcharge?")],
        # Refinement-style delta query, mirrors the guide's Example 2
        [SubQuery(id="q4", text="international travel reimbursement exception approval", intent_type="constraint")],
    ]

    for sub_queries in scenarios:
        print("=" * 72)
        print("Sub-queries:", [sq.text for sq in sub_queries])
        results = retriever.search(sub_queries)
        for r in results:
            preview = r.text[:90].replace("\n", " ")
            print(
                f"  {r.citation():22s} rrf={r.rrf_score:.4f} "
                f"dense={r.dense_score:.3f} sparse={r.sparse_score:.3f}  {preview}..."
            )
        print()


if __name__ == "__main__":
    main()
