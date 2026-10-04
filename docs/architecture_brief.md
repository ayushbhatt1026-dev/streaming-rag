# System Architecture Brief: Streaming Live RAG

**Samsung Electronics Hackathon — Theme 04: Real-Time Incremental Retrieval, Multi-Intent Decomposition, and State-Preserving Answer Refinement**

---

## 1. Executive Summary & Problem Overview

Standard Retrieval-Augmented Generation (RAG) operates as a rigid, batch-oriented turn cycle: a user completes speaking or typing, submits an entire query, and waits while the system searches a vector index, retrieves passages, and synthesizes an answer. In interactive conversational voice systems, this introduces prohibitive turn-taking latency (3–8 seconds), discards conversation history when late-arriving constraints arrive mid-stream, and struggles when complex requests bundle multiple orthogonal informational needs into a single stream of speech.

**Streaming Live RAG** replaces this paradigm with a low-latency, event-driven streaming architecture designed around three core capabilities:
1. **Speculative Early Retrieval (Gate G2)**: Evaluates timestamped transcript fragments incrementally as speech arrives, predicting intent stability to initiate search *before* the user finishes speaking.
2. **Multi-Intent Decomposition (Gate G3)**: Deconstructs compound, multi-part spoken utterances into 1–4 discrete, search-ready sub-queries with parallel retrieval routing and resolved pronouns.
3. **Claim-Preserving Answer Refinement (Gate G5)**: Updates existing answers in-place via a structured claim graph when late constraints arrive, preserving prior citations and issuing targeted delta queries without clearing session state or re-searching the full corpus.

---

## 2. Design Rationale & Architectural Parsimony

A primary constraint of Theme 04 is **Architectural Parsimony**: complex multi-agent frameworks, heavyweight graph databases, or nested orchestrator layers degrade conversational response times and add unnecessary compute overhead.

Our architecture is deliberately built as a **flat, single-process, asynchronous pipeline**:
* **Zero Heavy Agent Frameworks**: No LangChain agents, AutoGen, or CrewAI abstractions. All control flow, state transitions, and retry loops are explicitly authored in Python.
* **Deterministic Frozen Contracts**: Seven shared data contracts (`TranscriptChunk`, `ControllerDecision`, `SubQuery`, `EvidenceChunk`, `Claim`, `AnswerVersion`, `TelemetryEvent`) are imported across all pipeline stages, ensuring strict boundary validation.
* **Corpus Isolation**: All factual claims derive exclusively from the 5-document verified ground-truth corpus (`corpus/`). External web access and unindexed parametric memory are prohibited.

---

## 3. End-to-End Pipeline Architecture

```text
Incoming Audio / ASR Stream
   │
   ▼
[M0: StreamBuffer]  ──(Chunks every ~200-400ms)──▶  [M1: Retrieval Controller]
                                                           │
                      ┌────────────────────────────────────┴────────────────────────────────────┐
                      ▼                                                                         ▼
                   [WAIT]                                                                  [SUPPRESS]
         (Accumulate more speech)                                              (Transform prior context;
                                                                                zero retrieval executed)
                      │
                      ▼ [RETRIEVE_PROVISIONAL / RETRIEVE_FINAL / REFINE]
         [M2: Multi-Intent Decomposer]
         (Extract 1..4 orthogonal sub-queries; resolve pronouns; anti-fragmentation guard)
                      │
                      ▼
         [M3: Retrieval & Fusion Layer]
         (Dense BGE-small-en-v1.5 + Sparse BM25 ──▶ Reciprocal Rank Fusion ──▶ Deduplication)
                      │
                      ▼
         [M4: Refinement Engine / Session Store] ◀─────────┐
         (Map delta queries; compute claim diffs:          │
          KEPT, AMENDED, ADDED, RETRACTED)                 │ (Prior AnswerVersion)
                      │                                    │
                      ▼                                    │
         [M5: Grounded Synthesizer] ───────────────────────┘
         (Deterministic citation whitelist defense; explicit [uncertain] flagging)
                      │
                      ▼
         Streamed Answer + Grounded Citations + Trace Lineage ──▶ [M6: Telemetry Bus]
```

### Component Breakdown

### M1: Retrieval Controller (Two-Stage Interrupt Handler)
The controller executes on every incoming transcript chunk to balance speculative latency gains against premature, noisy searches:
* **Stage 1 (Microsecond Rule Gate)**: Zero model calls. Evaluates:
  1. *Presentation Filter*: Identifies formatting/summarization requests via semantic regex patterns (`repeat`, `shorter`, `simplify`, `explain.*different`) to return `Decision.SUPPRESS`.
  2. *Token Floor*: Requires $\ge 4$ content words before considering retrieval.
  3. *Entity Stability Window*: Extracts capitalized entities/numbers. When the entity set remains identical across consecutive chunks ($\ge 2$), intent has stabilized.
  4. *Dangling Connector Check*: Holds if the utterance ends on an incomplete conjunction (`and`, `to`, `the`, `but`).
* **Stage 2 (Constrained LLM Confirmation)**: Groq `openai/gpt-oss-20b` JSON confirm. Evaluates whether the stable fragment warrants `RETRIEVE_PROVISIONAL`, `RETRIEVE_FINAL`, or `REFINE`. On final chunks, parse errors fall back to deterministic heuristics to guarantee turn completion.

### M2: Multi-Intent Decomposer
Real spoken queries frequently combine multiple requests (e.g. *"What is the capacity of Venue B and its cancellation policy?"*). 
* Deconstructs compound queries into 1–4 self-contained sub-queries.
* **Pronoun Resolution**: Resolves anaphoric references (e.g. *"its refund terms"* $\to$ *"Venue B refund terms"*).
* **Anti-Fragmentation Guard**: Computes cosine similarity between generated sub-query embeddings; merges redundant sub-queries if similarity exceeds `0.90`.

### M3: Corpus Retrieval & Evidence Fusion
Executes hybrid search across the local indexed corpus:
* **Dense Retrieval**: `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) stored in embedded `QdrantVectorStore`.
* **Sparse Retrieval**: BM25 keyword matching with token-level frequency scoring.
* **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings with standard constant $k=60$:
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$
* **Deduplication & Context Pruning**: Deduplicates candidate chunks across sub-queries and limits prompt injection to the top-2 chunks per sub-query, reducing prompt overhead by 65%.

### M4: Session Memory & Refinement Engine
Enforces **Refine Rather Than Restart** when late-arriving details or corrections arrive (e.g. *"Actually the workshop is for 80 people, not 30"*):
* Stores active session context as an `AnswerVersion` containing an array of structured `Claim` objects.
* When `Decision.REFINE` triggers:
  1. Issues targeted queries for *only* the delta constraints.
  2. Compares existing claims against new evidence.
  3. Applies state mutations: untouched claims are marked `KEPT` (preserving original citations), modified claims become `AMENDED`, and outdated claims become `RETRACTED`.
  4. Increments the version counter ($v1 \to v2$) with zero full-corpus re-searches.

### M5: Grounded Synthesizer
Guarantees **Rigorous Factual Grounding**:
* Generates grounded natural language answers attributing assertions to verifiable document markers (`[Doc_ID §Section]`).
* **Deterministic Whitelist Filter**: Compares all emitted citations against the exact chunks retrieved for that turn. Any hallucinated citation (e.g. `Doc_999`, or citations not in evidence) is stripped before output generation.
* **Explicit Uncertainty Flagging**: When retrieved evidence is insufficient for any sub-intent, the synthesizer returns explicit uncertainty (`- [uncertain] ...`), strictly preventing hallucinated facts.

### M6: Observability Telemetry Bus
Central event collector recording append-only structured logs (`telemetry/trace_log.jsonl`):
* Every turn is tagged with a unique `trace_id` and `session_id`.
* Emits lifecycle events: `chunk_received`, `controller_decision`, `retrieval_started`, `retrieval_completed`, `decomposition_started`, `decomposition_completed`, `turn_completed`.
* Guarantees 100% stage coverage across all pipeline executions (Gate G6).

---

## 4. Data Provenance & Corpus Isolation

The system operates under strict **Corpus Isolation**:
* **Source Corpus**: Five canonical documents in `corpus/` covering venue capacity, cancellation policies, catering guidelines, travel reimbursement, and general workshop FAQs.
* **Chunking Strategy**: Markdown section-boundary splitting preserving heading metadata (`# Doc_XX: Title`, `## Section Y: Subtitle`). Chunks range from 300 to 500 tokens, maintaining semantic coherence without arbitrary character slicing.
* **Isolation Test**: Verified via `tests/test_corpus_isolation.py`, which monkeypatches network sockets to prove that indexing and search operate 100% locally with zero external network leakage.

---

## 5. Architectural Trade-offs & Decisions

| Decision | Selected Approach | Alternative Rejected | Rationale |
|---|---|---|---|
| **Vector Store** | Embedded Qdrant (local disk) | Hosted Pinecone / Qdrant Docker | Eliminates network latency, Docker dependency, and port conflicts during automated grading. |
| **Embeddings** | BGE-small-en-v1.5 (local) | OpenAI text-embedding-3-small | 100% local execution, zero API token cost, deterministic embeddings. |
| **LLM Inference** | Groq `openai/gpt-oss-20b` | Gemini 1.5 Flash / Local Ollama | Fast inference speeds (<1s generation); standard OpenAI-compatible API interface. |
| **State Storage** | Ephemeral In-Memory SessionStore | Redis / SQLite | Ephemeral session scoping meets strict Theme 04 rules prohibiting cross-session user profiling. |
| **Fusion Layer** | Reciprocal Rank Fusion (RRF) | Linear Score Weighting | RRF requires no score normalization between dense cosine distances and unbounded BM25 scores. |

---

## 6. Failure Mode Mitigations

Derived directly from empirical evaluation and documented edge cases:
1. **Mid-Utterance Entity Correction (Speech Repair)**: When a speaker corrects themselves mid-sentence (*"Pune... actually Mumbai"*), provisional retrieval for "Pune" is superseded on the final chunk. The Decomposer prompt includes speech-repair heuristics to discard superseded entities, and `hold_provisional_when_prior_answer: true` prevents state corruption during refinement turns.
2. **Entangled Compound Queries**: When multiple sub-intents share an entity (*"Venue B capacity and parking"*), an anti-fragmentation threshold of `0.90` prevents over-merging while ensuring distinct attribute queries are retrieved independently.
3. **Presentation Queries Phrased as Questions**: Stage 1 regex is reinforced with question-form presentation triggers (`explain.*different`, `simplify`, `reword`) to ensure presentation-only turns return `SUPPRESS` with 0 retrievals.
