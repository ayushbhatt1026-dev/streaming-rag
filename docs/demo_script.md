# System Demonstration Video Plan & Script (<= 5 Minutes)

**Samsung Electronics Hackathon — Theme 04: Streaming Live RAG**

---

## Video Specifications
* **Duration**: Exactly 4 minutes 30 seconds ($\le$ 5 minutes maximum).
* **Format**: Split screen — Live terminal execution / transcript streamer on the left, telemetry inspector and claim graph diff viewer on the right.
* **Key Visual Moments to Emphasize**:
  1. The **provisional search trigger** firing while speech text is still streaming.
  2. The **$v1 \to v2$ claim graph diff** showing untouched claims preserved with original IDs.
  3. The **"0 retrievals" suppression shot** when rephrasing or summarizing.

---

## Beat-by-Beat Video Script

### Beat 1: Problem Statement & Architectural Overview (0:00 – 0:45)
* **Visual**: Slide / diagram showing Batch RAG vs. Streaming Live RAG.
* **Audio / Script**:
  > *"Standard RAG systems wait until a user finishes speaking, creating unacceptable pauses in live voice interactions. They struggle with multi-intent requests and wipe conversation memory when users clarify details.
  > Today, we present Streaming Live RAG: an event-driven system built on three graded capabilities: speculative early retrieval, multi-intent decomposition, and claim-preserving answer refinement — achieving 100% across all six evaluation gates."*

---

### Beat 2: Speculative Early Retrieval (Gate G2) (0:45 – 1:30)
* **Visual**: Run `demo_pipeline.py`. Type or stream the compound utterance:
  *"I need to plan a customer workshop in Pune for 30 people and need the cancellation policy and catering options."*
* **What to Highlight on Screen**:
  * Point to the terminal at chunk 4: `"in Pune for 30 people"`.
  * Show the controller emit `RETRIEVE_PROVISIONAL` while `is_final: false`.
  * Highlight the telemetry timestamp showing search started **800ms before the utterance finished**.
* **Audio / Script**:
  > *"Watch chunk 4: as soon as the geographic entity and headcount stabilize, our microsecond rule gate escalates to the controller and fires a provisional retrieval. Search begins speculatively before the user even finishes speaking, gaining an 800-millisecond head start on voice synthesis."*

---

### Beat 3: Multi-Intent Decomposition (Gate G3) (1:30 – 2:15)
* **Visual**: Show the terminal decomposing the final utterance into 3 distinct sub-queries:
  1. `"workshop venue options in Pune for 30 people"`
  2. `"cancellation policy and refund window"`
  3. `"catering options and vegetarian accommodations"`
* **What to Highlight on Screen**:
  * Point to parallel sub-query retrieval across BGE dense and BM25 sparse indices with Reciprocal Rank Fusion.
* **Audio / Script**:
  > *"Rather than bottlenecking on a single vague search, our decomposer isolates three orthogonal sub-intents, resolves pronouns, and dispatches parallel searches merged via Reciprocal Rank Fusion."*

---

### Beat 4: Claim-Preserving Refinement (Gate G5) (2:15 – 3:15)
* **Visual**: Provide the follow-up clarification turn:
  *"Actually the workshop will be held at Venue A instead of Venue B."*
* **What to Highlight on Screen**:
  * Show the controller detect `Decision.REFINE`.
  * Show the **Query Log**: *only* delta queries for Venue A are issued; **0 full-corpus queries**.
  * Show the **Claim Graph Diff**:
    * Claim 1 (Capacity for Venue B): `STATUS: AMENDED` $\to$ updated to Venue A capacity.
    * Claim 2 (General cancellation policy): `STATUS: KEPT` $\to$ **Claim ID and citation preserved untouched**.
    * Answer version increments: $v1 \to v2$.
* **Audio / Script**:
  > *"Here is the critical differentiator: when the user introduces a late correction, traditional RAG discards context or restarts. Our system recognizes a refinement, queries only for the delta, and mutates the claim graph in-place. Claim 2 for cancellation terms remains completely untouched with its original citation, while the venue claim is updated in Version 2."*

---

### Beat 5: Query Suppression & Zero Retrieval (3:15 – 3:45)
* **Visual**: User says: *"Please repeat the previous answer in two concise bullet points."*
* **What to Highlight on Screen**:
  * Terminal displays `decision: SUPPRESS`, `n_retrievals_fired: 0`.
  * Show answer formatted into two bullets using existing session memory with zero database queries.
* **Audio / Script**:
  > *"When the user asks for a summary or reformat, the Stage 1 rule gate suppresses retrieval in under 1 millisecond. Zero queries are executed, preventing index drift and eliminating unnecessary token costs."*

---

### Beat 6: Strict Grounding & Explicit Uncertainty (Gate G4) (3:45 – 4:15)
* **Visual**: Ask an out-of-corpus probe: *"What is the pet policy for dogs at Venue A?"*
* **What to Highlight on Screen**:
  * Whitelist filter strips any hallucinated citations.
  * System outputs: `- [uncertain] Pet accommodations for Venue A are not documented in the corpus.`
* **Audio / Script**:
  > *"Our deterministic citation whitelist verifies every citation against retrieved evidence. When documentation is missing, the synthesizer emits an explicit uncertainty indicator rather than hallucinating an answer."*

---

### Beat 7: Full Gate Scorecard (G1–G6) & Conclusion (4:15 – 4:30)
* **Visual**: Display the master scorecard summary table showing **100% across all six gates**.
* **Audio / Script**:
  > *"In summary, Streaming Live RAG delivers sub-second incremental retrieval, precise multi-intent routing, and robust state continuity, passing all six automated evaluation gates with zero mock fallbacks. Thank you."*
