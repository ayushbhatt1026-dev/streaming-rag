# Telemetry & Observability Schema

**Samsung Electronics Hackathon — Theme 04: Streaming Live RAG**

---

## 1. Overview & Observability Principles

The **M6 Telemetry Bus** provides end-to-end auditability across the streaming pipeline. Every incoming chunk, routing decision, retrieval query, claim mutation, and synthesized answer is recorded as an immutable, structured event in `telemetry/trace_log.jsonl`.

### Key Design Principles:
* **Trace Isolation**: Every user turn is tagged with a globally unique `trace_id` and `session_id`.
* **Sub-Millisecond Non-Blocking Emission**: Events are appended directly to file storage without blocking pipeline execution.
* **100% Stage Coverage (Gate G6)**: Every trace records events across all required stages: `stream_buffer`, `controller`, `retrieval`, and `synthesis`.

---

## 2. Core Event Schema

Every emitted event conforms strictly to the `TelemetryEvent` data contract in [`contracts.py`](file:///C:/Users/ayush/Desktop/try1/streaming-live-rag/contracts.py):

```python
@dataclass
class TelemetryEvent:
    trace_id: str             # Unique identifier for the single turn
    session_id: str           # Session identifier across multi-turn dialogs
    stage: str                # Component: stream_buffer | controller | decomposer | retrieval | synthesis
    event: str                # Action name
    t_ms: float               # Relative timestamp within the utterance stream (ms)
    payload: Dict[str, Any]   # Event-specific attributes
```

---

## 3. Event Catalog & Payload Specifications

### 1. `chunk_received` (`stage: "stream_buffer"`)
Emitted as each incremental ASR fragment arrives at the ingestion buffer.
```json
{
  "trace_id": "t-7a8b9c0d",
  "session_id": "sess-01",
  "stage": "stream_buffer",
  "event": "chunk_received",
  "t_ms": 520.0,
  "payload": {
    "seq": 2,
    "text_delta": "for 30 people",
    "is_final": false
  }
}
```

### 2. `controller_decision` (`stage: "controller"`)
Emitted when the M1 Retrieval Controller evaluates an incoming chunk.
```json
{
  "trace_id": "t-7a8b9c0d",
  "session_id": "sess-01",
  "stage": "controller",
  "event": "controller_decision",
  "t_ms": 520.0,
  "payload": {
    "decision": "RETRIEVE_PROVISIONAL",
    "confidence": 0.85,
    "reason": "entity_stable",
    "latency_ms": 0.12,
    "utterance_progress_pct": 60.0,
    "buffer_text": "I need a workshop in Pune for 30 people"
  }
}
```

### 3. `decomposition_started` & `decomposition_completed` (`stage: "decomposer"`)
Emitted when compound utterances are split into parallel sub-queries.
```json
{
  "trace_id": "t-7a8b9c0d",
  "session_id": "sess-01",
  "stage": "decomposer",
  "event": "decomposition_completed",
  "t_ms": 520.0,
  "payload": {
    "n_sub_queries": 2,
    "sub_queries": [
      "workshop venue options in Pune for 30 people",
      "cancellation terms and refund policies"
    ]
  }
}
```

### 4. `retrieval_started` & `retrieval_completed` (`stage: "retrieval"`)
Emitted when the search engine executes dense + sparse retrieval.
```json
{
  "trace_id": "t-7a8b9c0d",
  "session_id": "sess-01",
  "stage": "retrieval",
  "event": "retrieval_completed",
  "t_ms": 535.4,
  "payload": {
    "trigger": "provisional",
    "n_candidates": 8,
    "top_doc_ids": ["Doc_01", "Doc_02"]
  }
}
```

### 5. `turn_completed` (`stage: "synthesis"`)
Emitted upon final answer generation and citation verification.
```json
{
  "trace_id": "t-7a8b9c0d",
  "session_id": "sess-01",
  "stage": "synthesis",
  "event": "turn_completed",
  "t_ms": 1420.0,
  "payload": {
    "answer_preview": "- Riverside Hall supports up to 40 seated [Doc_01 §Section 1]...",
    "version": 1,
    "n_claims": 2,
    "n_uncertainties": 0,
    "n_fabricated_stripped": 0
  }
}
```

---

## 4. Derived Observability Metrics

The telemetry bus provides real-time computation of key operational metrics:

| Metric | Derivation Formula | Purpose |
|---|---|---|
| **Early Retrieval Delta ($\Delta t_{early}$)** | $t_{final\_chunk} - t_{retrieval\_started}$ | Quantifies the head start (ms) gained by speculative search |
| **Stage Decision Latency** | $t_{decision\_end} - t_{chunk\_arrival}$ | Monitors controller overhead (target < 5ms for Stage 1) |
| **Time to First Token (TTFT)** | $t_{first\_token} - t_{synthesis\_start}$ | Tracks generation latency for conversational responsiveness |
| **Citation Whitelist Violations** | $\sum n_{fabricated\_stripped}$ | Verifies zero hallucinated citations reach user output |
| **Estimated Turn Cost** | $(N_{tokens\_prompt} \times \$0.15 + N_{tokens\_gen} \times \$0.60) / 10^6$ | Inference cost tracking per conversational turn |
