"""
Interactive CLI for Streaming Live RAG.

Usage:
  # 1. Interactive multi-turn chat session:
  python chat.py

  # 2. Ask a one-off question directly from the command line:
  python chat.py "What is the seated capacity of Venue B and its cancellation policy?"

Commands in interactive mode:
  'new'   - Start a fresh session (clears conversation state)
  'state' - Inspect current active AnswerVersion and claim graph
  'exit'  - Quit the application
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Optional

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from contracts import Decision
from controller.controller import RetrievalController
from decomposer.backends import GroqDecomposeBackend, HeuristicDecomposeBackend
from decomposer.decomposer import MultiIntentDecomposer
from llm.groq_client import GroqLLMClient, _load_env_file_if_needed
from orchestrator.walking_skeleton import run_one_utterance
from refinement.refinement_engine import RefinementEngine
from retrieval.embeddings import BGEEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import QdrantVectorStore
from session.store import SessionStore
from synthesis.grounded_synthesizer import GroundedSynthesizer
from telemetry.bus import TelemetryBus


def print_banner() -> None:
    print("=" * 72)
    print("  STREAMING LIVE RAG — Interactive Conversational Console")
    print("  Theme 04: Incremental Retrieval | Multi-Intent | Claim Refinement")
    print("=" * 72)


def main() -> None:
    _load_env_file_if_needed()
    print_banner()

    print("[1/3] Initializing local neural embeddings (BGE-small-en-v1.5)...")
    emb = BGEEmbeddingBackend()

    print("[2/3] Connecting to Qdrant vector store and indexing corpus/...")
    vstore = QdrantVectorStore()
    retriever = Retriever(embedding_backend=emb, vector_store=vstore)
    n_chunks = retriever.index_corpus(corpus_dir="corpus")
    print(f"      Indexed {n_chunks} chunks from corpus/ (100% Corpus Isolation).")

    print("[3/3] Initializing Groq LLM client (openai/gpt-oss-20b)...")
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    if has_groq:
        groq_client = GroqLLMClient()
        from controller.stage2 import GroqStage2Confirm
        stage2_backend = GroqStage2Confirm(client=groq_client)
        decomp_backend = GroqDecomposeBackend(client=groq_client)
        synthesizer = GroundedSynthesizer(llm_client=groq_client)
        print("      Connected to Groq LLM (Production Stack Active).")
    else:
        from controller.stage2 import HeuristicStage2Confirm
        stage2_backend = HeuristicStage2Confirm()
        decomp_backend = HeuristicDecomposeBackend()
        from synthesis.naive_synthesizer import NaiveSynthesizer
        synthesizer = NaiveSynthesizer()
        print("      [Warning] GROQ_API_KEY not found in api_key_env.env; using offline heuristic mode.")

    session_store = SessionStore()
    session_id = f"cli-{uuid.uuid4().hex[:6]}"
    telemetry = TelemetryBus(jsonl_path="telemetry/trace_log.jsonl")

    # If question passed via command-line arguments, run it directly and exit
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        print(f"\nUser Query: {question}\n")
        _process_turn(question, stage2_backend, decomp_backend, emb, retriever,
                      synthesizer, session_store, session_id, telemetry)
        return

    print("\nSystem ready! Type your question, clarification, or refinement below.")
    print("Commands: '/mic' (speak into microphone), 'new' (reset session), 'state' (view claims), 'exit' (quit).\n")

    prior_utterance: Optional[str] = None
    mic_streamer = None

    while True:
        try:
            prompt_str = f"[{session_id}] You (or '/mic' to speak): "
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("Session ended. Traces saved to telemetry/trace_log.jsonl. Goodbye!")
            break

        if cmd == "new":
            session_id = f"cli-{uuid.uuid4().hex[:6]}"
            prior_utterance = None
            print(f"\nStarted new session: {session_id} (state cleared).\n")
            continue

        if cmd == "state":
            av = session_store.get(session_id)
            if av is None:
                print("\nNo active AnswerVersion in this session yet.\n")
            else:
                print(f"\n--- Session State: AnswerVersion {av.version} ---")
                for i, cl in enumerate(av.claims):
                    print(f"  [{cl.status.name}] Claim {i+1} ({cl.id}): {cl.text}")
                    print(f"         Citations: {', '.join(cl.citations) if cl.citations else 'none'}")
                if av.uncertainties:
                    print(f"  Uncertainties: {av.uncertainties}")
                print("------------------------------------------\n")
            continue

        # Audio Device Listing command
        if cmd in ("/devices", "devices", "/mic-devices"):
            try:
                from audio_streamer import list_microphones
                devs = list_microphones()
                print("\n--- Available Audio Input Devices ---")
                for d in devs:
                    tag = " [SYSTEM DEFAULT]" if d["is_default"] else ""
                    print(f"  [{d['index']}] {d['name']} (channels: {d['channels']}){tag}")
                print("--------------------------------------")
                print("To use a specific device, type: /mic <id> (e.g., '/mic 1')\n")
            except Exception as e:
                print(f"Error querying devices: {e}")
            continue

        # Option 1: Live Voice Input via Microphone
        if cmd.startswith("/mic") or cmd.startswith("mic") or cmd.startswith("/voice"):
            try:
                from audio_streamer import MicrophoneStreamer, AUDIO_AVAILABLE
                if not AUDIO_AVAILABLE:
                    print("\n[Notice] Audio packages missing. Install: pip install sounddevice faster-whisper\n")
                    continue

                # Parse optional device index: e.g. "/mic 2"
                parts = user_input.split()
                target_dev = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

                if mic_streamer is None or target_dev is not None:
                    mic_streamer = MicrophoneStreamer(device_index=target_dev)

                # Stream audio and accumulate transcript
                chunks = list(mic_streamer.listen_and_stream(session_id))
                spoken_text = (mic_streamer.last_full_transcript or " ".join(c.text_delta for c in chunks)).strip()

                if not spoken_text:
                    print("  (No speech was transcribed. Try speaking closer to the mic or type '/devices'.)\n")
                    continue

                print(f"\n🎙️ [Transcribed Question]: \"{spoken_text}\"")
                prior_utterance = _process_turn(
                    spoken_text, stage2_backend, decomp_backend, emb, retriever,
                    synthesizer, session_store, session_id, telemetry,
                    prior_utterance=prior_utterance
                )
            except Exception as e:
                print(f"\n[Microphone Error]: {e}\n")
            continue

        # Execute text turn
        prior_utterance = _process_turn(
            user_input, stage2_backend, decomp_backend, emb, retriever,
            synthesizer, session_store, session_id, telemetry,
            prior_utterance=prior_utterance
        )


def _process_turn(
    utterance: str,
    stage2_backend,
    decomp_backend,
    embedding_backend,
    retriever: Retriever,
    synthesizer,
    session_store: SessionStore,
    session_id: str,
    telemetry: TelemetryBus,
    prior_utterance: Optional[str] = None,
) -> str:
    trace_id = f"t-{uuid.uuid4().hex[:8]}"

    # Prior answer context for M1 controller (Step 7 refinement detection)
    prior_av = session_store.get(session_id)
    prior_context = prior_utterance if prior_av is not None else None

    ctrl = RetrievalController(
        stage2=stage2_backend,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
        prior_context=prior_context,
    )

    decomp = MultiIntentDecomposer(
        backend=decomp_backend,
        embedding_backend=embedding_backend,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
    )

    ref_engine = RefinementEngine(
        decomposer=decomp,
        retriever=retriever,
        embedding_backend=embedding_backend,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
    )

    print("  -> Processing streaming speech chunks...")
    result = run_one_utterance(
        utterance=utterance,
        controller=ctrl,
        decomposer=decomp,
        retriever=retriever,
        synthesizer=synthesizer,
        telemetry=telemetry,
        session_id=session_id,
        trace_id=trace_id,
        session_store=session_store,
        refinement_engine=ref_engine,
    )

    # Display Routing Decision & Details
    dec_val = result.final_decision.value if result.final_decision else "UNKNOWN"
    early_str = "YES (speculative search triggered before utterance end)" if result.early_retrieval else "NO"
    print(f"  -> Decision: [{dec_val}] | Early Retrieval: {early_str}")

    if result.sub_query_texts:
        print(f"  -> Decomposed Sub-Queries ({len(result.sub_query_texts)}):")
        for sq in result.sub_query_texts:
            print(f"       • {sq}")

    if result.final_decision == Decision.REFINE and result.answer_version:
        print(f"  -> [REFINE] Mutated Claim Graph in-place (AnswerVersion {result.answer_version.version})")
        for cl in result.answer_version.claims:
            print(f"       [{cl.status.name}] {cl.text[:70]}... ({', '.join(cl.citations)})")

    # Display Final Answer
    print("\n" + "-" * 72)
    print(f"ANSWER (trace: {trace_id}):")
    print(result.answer.strip())
    print("-" * 72 + "\n")

    return utterance


if __name__ == "__main__":
    main()
