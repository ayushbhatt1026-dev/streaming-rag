"""
Runnable, offline-safe demo of Steps 4-6: the walking skeleton with the
REAL Controller (M1) and Decomposer (M2) wired in — using their offline
heuristic backends (HeuristicStage2Confirm, HeuristicDecomposeBackend) so
this runs with zero API key and zero internet.

For your actual submission, switch the two lines marked PRODUCTION below
to the Gemini-backed classes, exactly like demo_retrieval.py's BGE/Qdrant
swap.

Run: python demo_pipeline.py
"""

import uuid

from controller.controller import RetrievalController
from controller.stage2 import HeuristicStage2Confirm  # PRODUCTION: GeminiStage2Confirm
from decomposer.backends import HeuristicDecomposeBackend  # PRODUCTION: GeminiDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from orchestrator.walking_skeleton import run_one_utterance
from retrieval.embeddings import TfidfEmbeddingBackend  # PRODUCTION: BGEEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import InMemoryVectorStore  # PRODUCTION: QdrantVectorStore
from synthesis.naive_synthesizer import NaiveSynthesizer
from telemetry.bus import TelemetryBus

SCENARIOS = [
    # Single-intent -> decomposer should emit exactly one sub-query, no padding
    "What is the seated capacity of Venue B?",
    # Compound, mirrors the guide's own Example 1 -> should split into ~3 sub-queries
    "I need to plan a customer workshop in Pune for 30 people and I need the cancellation policy and the catering options",
    # Presentation-only -> should SUPPRESS, zero retrievals
    "Can you repeat that in two bullets please",
    # Unanswerable -> should retrieve, but say "not documented"
    "Is there a weekend booking surcharge",
]


def main() -> None:
    embedding_backend = TfidfEmbeddingBackend()

    retriever = Retriever(embedding_backend=embedding_backend, vector_store=InMemoryVectorStore())
    n_chunks = retriever.index_corpus(corpus_dir="corpus")
    print(f"Indexed {n_chunks} chunks from corpus/\n")

    telemetry = TelemetryBus(jsonl_path="telemetry/trace_log.jsonl")
    synthesizer = NaiveSynthesizer()

    for utterance in SCENARIOS:
        session_id = f"demo-{uuid.uuid4().hex[:6]}"
        trace_id = str(uuid.uuid4())

        controller = RetrievalController(
            stage2=HeuristicStage2Confirm(),
            telemetry=telemetry,
            session_id=session_id,
            trace_id=trace_id,
        )
        decomposer = MultiIntentDecomposer(
            backend=HeuristicDecomposeBackend(),
            embedding_backend=embedding_backend,
            telemetry=telemetry,
            session_id=session_id,
            trace_id=trace_id,
        )

        print("=" * 72)
        print(f"Utterance : {utterance!r}")
        print(f"trace_id  : {trace_id}")

        result = run_one_utterance(
            utterance, controller, decomposer, retriever, synthesizer,
            telemetry, session_id, trace_id,
        )

        print(f"Final decision      : {result.final_decision}")
        print(f"Early retrieval?     : {result.early_retrieval}  (fired before is_final)")
        print(f"Retrievals fired     : {result.n_retrievals_fired}")
        print(f"Sub-queries          : {result.sub_query_texts}")
        print("Answer:")
        print(result.answer)
        print(f"\nStage coverage for this trace: {telemetry.stage_coverage(trace_id)}")
        print()


if __name__ == "__main__":
    main()
