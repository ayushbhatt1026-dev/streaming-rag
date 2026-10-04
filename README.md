# Streaming Live RAG — Samsung Electronics Hackathon (Theme 04)

Real-Time Incremental Retrieval, Multi-Intent Decomposition, and State-Preserving Answer Refinement.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker Compliant](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gate Scorecard: 100% PASS](https://img.shields.io/badge/Scorecard-100%25%20PASS-brightgreen.svg)](docs/benchmarking_report.md)

DEMO Video : https://drive.google.com/file/d/1yaAHiJ2yJ5CDpkUIMHp2NWVsBNPywo2F/view?usp=drivesdk

---

## 1. Overview & Architecture

Standard RAG operates in static batch mode: wait for the user to finish speaking, search, generate. In contrast, **Streaming Live RAG** processes speech incrementally chunk-by-chunk and implements three primary graded behaviors:

1. **Speculative Early Retrieval (Gate G2)**: Senses when intent and entities have stabilized mid-utterance and initiates retrieval *before* the speaker finishes.
2. **Multi-Intent Decomposition (Gate G3)**: Splits compound spoken requests into 1–4 orthogonal, self-contained sub-queries with resolved pronouns and anti-fragmentation guards.
3. **Claim-Preserving Refinement (Gate G5)**: Updates existing answer versions in-place using a structured claim graph and delta search, preserving prior valid citations and claims without restarting.

```text
Streamed transcript chunks (every ~200-400ms from microphone or simulator)
   → M1 Retrieval Controller (Rule Gate + Stage 2 Confirm: WAIT / RETRIEVE / SUPPRESS / REFINE)
   → [if Retrieve] M2 Multi-Intent Decomposer (1..4 orthogonal sub-queries)
   → M3 Retrieval & Fusion (Dense BGE-small + Sparse BM25 → Reciprocal Rank Fusion)
   → M4 Session-Bound Claim Refinement (In-place mutation to AnswerVersion 2+)
   → M5 Grounded Synthesizer (Claim Graph, Citation Whitelist Defense, Explicit Uncertainty)
   → Streamed answer + citations + trace_id (100% Telemetry Coverage to M6 Bus)
```

---

## 2. Quickstart & Usage

### Option A: Local Run (Clean CLI Runner)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/streaming-live-rag.git
   cd streaming-live-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key (Groq)**:
   Copy the example environment template to create `api_key_env.env`:
   ```bash
   cp api_key_env.env.example api_key_env.env
   ```
   Edit `api_key_env.env` to add your Groq API key:
   ```env
   GROQ_API_KEY="gsk_..."
   ```

4. **Launch Interactive Console**:
   ```bash
   # Continuous multi-turn session with live text or microphone speech:
   python chat.py

   # Or execute a one-off query straight from the command line:
   python chat.py "What is the seated capacity of Venue B and its cancellation policy?"
   ```

5. **Commands inside `chat.py`**:
   - Type `/mic` to record via your microphone (watch the live VU meter and press `Enter` when done).
   - Type `/devices` to list audio input devices (or `/mic <id>` to switch microphones).
   - Type `state` to inspect the active claim graph and `AnswerVersion`.
   - Type `new` to start a fresh session (clears conversation state).
   - Type `exit` to quit.

---

### Option B: Docker Compose (Gate G1 Containerization)

Run the containerized application on any clean machine:
```bash
docker compose up --build
```
Verify the health check endpoint:
```bash
curl http://localhost:8000/health
```

---

## 3. Evaluation Gates Scorecard (G1 – G6)

The system is evaluated against the 6 gates from the Samsung Theme 04 specification on a 30% held-out test split:

| Gate | Criterion | Target Threshold | Achieved (30% Split) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | **Reproducibility** | Pass/Fail | Clean Docker build & automated test pass | **100% PASS** |
| **G2** | **Early Retrieval** | $\ge 80\%$ eligible queries | 3/3 held-out items triggered early retrieval | **100% PASS** |
| **G3** | **Multi-Intent Identification** | $\ge 70\%$ compound queries | 8/8 compound queries decomposed accurately | **100% PASS** |
| **G4** | **Factual Grounding** | $\ge 85\%$ support, 0 hallucinations | 100% citation support, 0 fabricated citations | **100% PASS** |
| **G5** | **Session Refinement** | Verified state continuity | In-place claim mutation, delta-only query | **100% PASS** |
| **G6** | **Telemetry Observability** | 100% trace coverage | 5/5 traces emitted full lifecycle events | **100% PASS** |

Run the automated scorecard evaluator:
```bash
python eval/run_scorecard.py
```

Run the architectural ablations:
```bash
python eval/run_ablations.py
```

Run the full pytest suite (142 unit & integration tests):
```bash
python -m pytest tests/ -v
```

---

## 4. Repository Structure

```text
streaming-live-rag/
├── app/
│   ├── Dockerfile                 # G1 Docker container definition
│   ├── main.py                    # FastAPI server with /health and /process endpoints
│   └── requirements.txt           # Container runtime dependencies
├── config/
│   ├── controller.yaml            # Token floor, stability window, escalation flags
│   ├── decomposer.yaml            # Max sub-queries, similarity thresholds
│   ├── refinement.yaml            # Touch classification & contradiction config
│   └── retrieval.yaml             # Hybrid weights, top-k, chunk overlap
├── controller/
│   ├── controller.py              # M1 Hierarchical 2-stage retrieval controller
│   ├── rule_gate.py               # Stage 1 deterministic interrupt handler (<0.1ms)
│   └── stage2.py                  # Stage 2 LLM confirm & heuristic fallbacks
├── corpus/                        # Isolated 5-document evaluation corpus
│   ├── Doc_01_Venue_Capacity_Directory.txt
│   ├── Doc_02_Event_Cancellation_and_Refund_Policy.txt
│   ├── Doc_03_Catering_and_Dietary_Guidelines.txt
│   ├── Doc_04_Pune_Corporate_Workshop_Facilities.txt
│   └── Doc_05_Travel_Expense_and_Reimbursement_Policy.txt
├── decomposer/
│   ├── backends.py                # LLM & heuristic decomposition backends
│   └── decomposer.py              # M2 Multi-intent query splitter & de-duplicator
├── docs/                          # Engineering Deliverables (Theme 04 Spec)
│   ├── architecture_brief.md      # <= 6-page comprehensive architecture brief
│   ├── benchmarking_report.md     # Gate scorecard, ablations, & failure post-mortems
│   ├── telemetry_schema.md        # TelemetryEvent & derived metrics schema
│   └── demo_script.md             # 4.5-minute video demonstration walkthrough script
├── eval/
│   ├── latency_profiler.py        # Latency breakdown per pipeline stage
│   ├── run_ablations.py           # Hybrid vs Dense & Rule vs LLM gate ablations
│   └── run_scorecard.py           # Automated evaluation runner for Gates G1-G6
├── llm/
│   ├── gemini_client.py           # Google Gemini API client
│   ├── groq_client.py             # Groq client (openai/gpt-oss-20b production stack)
│   └── openai_client.py           # OpenAI API client
├── orchestrator/
│   └── walking_skeleton.py        # Streaming orchestrator driving the pipeline
├── refinement/
│   ├── claim_builder.py           # Constructs initial claim graph from evidence
│   ├── contradiction_checker.py   # Flags incompatible prior assertions
│   ├── refinement_engine.py       # M4 In-place claim mutator (AnswerVersion 2+)
│   ├── session_store.py           # Session store adapter
│   └── touch_classifier.py       # Classifies touched vs untouched claims
├── retrieval/
│   ├── chunking.py                # Structural section-preserving chunker
│   ├── embeddings.py              # Local BGE-small-en-v1.5 embedding backend
│   ├── fusion.py                  # Reciprocal Rank Fusion (RRF)
│   ├── retriever.py               # M3 Hybrid Dense + BM25 retrieval engine
│   ├── sparse_bm25.py             # Custom BM25 implementation
│   └── vector_store.py            # Local Qdrant embedded vector database
├── session/
│   └── store.py                   # In-memory TTL ephemeral session store
├── simulator/
│   └── transcript_simulator.py    # Deterministic chunk streaming simulator
├── synthesis/
│   ├── grounded_synthesizer.py    # M5 Grounded synthesizer with claim extraction
│   └── naive_synthesizer.py       # Offline fallback synthesizer
├── telemetry/
│   └── bus.py                     # M6 High-throughput JSONL telemetry event bus
├── tests/                         # Full automated test suite (142 tests)
├── audio_streamer.py              # Live microphone streaming (Silero VAD + AGC)
├── chat.py                        # Interactive multi-turn CLI console
├── contracts.py                   # Core dataclasses and schemas
├── docker-compose.yml             # Single-command container deployment
├── requirements.txt               # Pinned package requirements
├── requirements.lock              # Fully pinned reproducible environment lockfile
└── README.md                      # Project documentation
```

---

## 5. Clean Repository Maintenance

To keep the repository clean and ready for public GitHub release, the following files have been deleted or git-ignored:

### Deleted Files (Safe to Remove)
The following intermediate development artifacts were superseded by `chat.py`, `eval/run_scorecard.py`, and the formal test suite:
- `validate_phase2_exit.py` (Superseded by Step 9 automated scorecard)
- `validate_step7.py` (Superseded by `tests/test_step7_refinement.py`)
- `demo_pipeline.py` (Superseded by `chat.py`)
- `demo_refinement.py` (Superseded by `chat.py` refinement handling)
- `demo_retrieval.py` (Superseded by `retrieval/retriever.py` test suite)
- `simulator/session/` (Redundant duplicate of `session/store.py`)

### Git-Ignored Files (Never Committed to GitHub)
The `.gitignore` file automatically protects sensitive credentials and transient runtime data:
- `api_key_env.env` (Contains your private `GROQ_API_KEY`)
- `qdrant_data/` (Local vector store SQLite cache generated at startup)
- `telemetry/*.jsonl` (Local runtime event logs generated during execution)
- `__pycache__/` and `*.pyc` (Compiled Python bytecode)
- `.venv/` and `*.log` (Local environments and debug logs)
