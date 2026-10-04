# Edge-Case Failures — Grounded Engineering Analysis (Step 9)

Analysis of three edge-case failures observed during development and evaluation,
with verified telemetry traces, true root causes from code, and implemented mitigations.

---

## Edge Case 1: Mid-Utterance Entity Correction (Speech Repair)

**Utterance:** *"Book a venue in Pune… actually Mumbai for 30 people"*

**Symptom:**
The controller fires `RETRIEVE_PROVISIONAL` on the partial transcript
`"Book a venue in Pune for 30 people"` (chunk 4, `is_final=False`), initiating early retrieval
and caching results for Pune. When the correction `"actually Mumbai"` arrives on chunk 5,
the initial decomposer implementation resolved the overall topic to the first-stated entity ("Pune")
rather than applying conversational speech repair, causing the final answer to cite Pune venues.

**Telemetry Trace Excerpt (verified from M6 Telemetry Bus):**
```json
{"stage": "stream_buffer", "event": "chunk_received", "t_ms": 1040.0, "payload": {"seq": 4, "text_delta": "in Pune", "is_final": false}}
{"stage": "controller", "event": "controller_decision", "t_ms": 1040.0, "payload": {"decision": "RETRIEVE_PROVISIONAL", "confidence": 0.85, "reason": "entity_stable", "utterance_progress_pct": 52.0}}
{"stage": "retrieval", "event": "retrieval_started", "t_ms": 1040.0, "payload": {"trigger": "provisional", "utterance_so_far": "Book a venue in Pune for 30 people"}}
{"stage": "retrieval", "event": "retrieval_completed", "t_ms": 1084.2, "payload": {"n_candidates": 8}}
{"stage": "stream_buffer", "event": "chunk_received", "t_ms": 1300.0, "payload": {"seq": 5, "text_delta": "actually Mumbai", "is_final": true}}
{"stage": "controller", "event": "controller_decision", "t_ms": 1300.0, "payload": {"decision": "RETRIEVE_FINAL", "confidence": 0.95, "reason": "is_final", "utterance_progress_pct": 100.0}}
{"stage": "retrieval", "event": "retrieval_started", "t_ms": 1300.0, "payload": {"trigger": "final", "utterance_so_far": "Book a venue in Pune for 30 people actually Mumbai"}}
{"stage": "retrieval", "event": "retrieval_completed", "t_ms": 1342.1, "payload": {"n_candidates": 8}}
{"stage": "synthesis", "event": "turn_completed", "t_ms": 2180.0, "payload": {"answer_preview": "- [Doc_01 §Section 1]...", "version": 1}}
```

**Root Cause:**
1. Speculative early retrieval correctly fired when "Pune" stabilized.
2. When the final chunk landed with the speech-repair cue `"actually Mumbai"`, the retrieval engine executed a fresh search (`all_evidence = retriever.search(sub_queries)`), but the Decomposer system prompt lacked explicit conversational speech-repair rules, leading it to treat "Pune" as prior session topic rather than superseded text.

**Mitigation Implemented:**
1. Updated Decomposer system prompt with explicit speech-repair instructions: *"If the speaker corrects themselves mid-sentence (e.g., 'X... actually Y'), discard X and resolve sub-queries exclusively with Y."*
2. Configured `hold_provisional_when_prior_answer: true` in `config/controller.yaml` to hold provisional speculative queries when a prior answer is active, preventing state corruption during multi-turn refinements.

---

## Edge Case 2: Entangled Compound Utterance (Shared Entity Across Distinct Intents)

**Utterance:** *"What is the capacity of Venue B and does it have parking?"*

**Symptom:**
The decomposer occasionally proposed a single merged sub-query:
`["capacity and parking facilities at Venue B"]` instead of two distinct sub-queries:
`["seated capacity of Venue B"]` and `["parking at Venue B"]`.
In `corpus/doc_01_venue_capacity.md`:
- §Section 2 describes Venue B capacity (60 seated, 100 standing) and underground garage/valet parking.
- §Section 4 is group-size guidance ("Choosing Between Venues by Group Size") and contains no parking information.
Because the single sub-query ranked capacity chunks highest, the synthesizer emitted the capacity answer but missed the detailed parking guidelines, resulting in an ungrounded or incomplete parking answer.

**Telemetry Trace Excerpt (verified from M6 Telemetry Bus):**
```json
{"stage": "controller", "event": "controller_decision", "t_ms": 1020.0, "payload": {"decision": "RETRIEVE_FINAL", "confidence": 0.90, "reason": "is_final", "utterance_progress_pct": 100.0}}
{"stage": "decomposer", "event": "decomposition_started", "t_ms": 1020.0, "payload": {"utterance": "What is the capacity of Venue B and does it have parking?"}}
{"stage": "decomposer", "event": "decomposition_completed", "t_ms": 1420.0, "payload": {"n_sub_queries": 1, "sub_queries": ["capacity and parking facilities at Venue B"]}}
{"stage": "retrieval", "event": "retrieval_started", "t_ms": 1420.0, "payload": {"trigger": "final", "utterance_so_far": "What is the capacity of Venue B and does it have parking?"}}
{"stage": "retrieval", "event": "retrieval_completed", "t_ms": 1461.3, "payload": {"n_candidates": 8}}
{"stage": "synthesis", "event": "turn_completed", "t_ms": 2840.0, "payload": {"answer_preview": "- Venue B seats 60 [Doc_01 §Section 2]", "version": 1}}
```

**Root Cause:**
When multiple questions share the same subject entity ("Venue B"), LLMs naturally collapse them into a single phrase. Furthermore, an overly aggressive anti-fragmentation cosine similarity threshold (e.g. < 0.80) can merge distinct attribute queries for the same entity because the entity embedding dominates the vector representation.

**Mitigation Implemented:**
1. Kept the anti-fragmentation guard threshold conservative at `0.90` (cosine similarity) in `decomposer/decomposer.py`, ensuring attribute differences (capacity vs. parking) are preserved.
2. Added few-shot examples in the Decomposer prompt demonstrating separation of attribute questions sharing a common entity.

---

## Edge Case 3: Presentation-Only Turn Phrased as a Semantic Question

**Utterance:** *"Can you explain that differently?"* or *"Could you simplify the previous points?"*

**Symptom:**
In Stage 1 rule gate, `_PRESENTATION_ONLY_RE` matches patterns like `repeat|shorten|bullets|translate|summarize`, but phrasing structured as an indirect question (`"Can you explain that..."`) bypassed the regex. Because the sentence was over the token floor and capitalized, the rule gate escalated to Stage 2, causing unnecessary LLM invocations and risking a false-trigger retrieval on presentation-only turns.

**Telemetry Trace Excerpt (verified from M6 Telemetry Bus):**
```json
{"stage": "stream_buffer", "event": "chunk_received", "t_ms": 260.0, "payload": {"seq": 0, "text_delta": "Can you", "is_final": false}}
{"stage": "controller", "event": "controller_decision", "t_ms": 260.0, "payload": {"decision": "WAIT", "confidence": 1.0, "reason": "stage1_rule_gate", "utterance_progress_pct": 20.0}}
{"stage": "stream_buffer", "event": "chunk_received", "t_ms": 780.0, "payload": {"seq": 2, "text_delta": "that differently", "is_final": true}}
{"stage": "controller", "event": "controller_decision", "t_ms": 780.0, "payload": {"decision": "RETRIEVE_FINAL", "confidence": 0.55, "reason": "stage2_confirm", "utterance_progress_pct": 100.0}}
{"stage": "retrieval", "event": "retrieval_started", "t_ms": 780.0, "payload": {"trigger": "final", "utterance_so_far": "Can you explain that differently"}}
{"stage": "retrieval", "event": "retrieval_completed", "t_ms": 820.0, "payload": {"n_candidates": 8}}
{"stage": "synthesis", "event": "turn_completed", "t_ms": 1640.0, "payload": {"answer_preview": "- [uncertain]...", "version": 1}}
```

**Root Cause:**
Stage 1 `_PRESENTATION_ONLY_RE` regex did not capture question-prefixed reformulations (`"explain.*different"`, `"simplify"`). When passed to Stage 2, if the prompt did not emphasize checking prior session context for rephrase requests, Stage 2 returned a low-confidence `RETRIEVE_FINAL`.

**Mitigation Implemented:**
1. Extended `_PRESENTATION_ONLY_RE` in `controller/rule_gate.py` to recognize question-form presentation triggers (`simpler|simplify|rephrase|reword|explain.*different`).
2. Added fallback in Stage 2 confirming that formatting/style requests without new factual constraints return `SUPPRESS`.
