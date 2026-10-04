"""
Chunk simulator (WS-B, Phase 1 Step 4).

Turns a plain sentence into a list of TranscriptChunk objects, 2 words at a
time, timed ~200-400ms apart with jitter — mimicking how a real streaming
ASR (speech-to-text) system would deliver partial transcripts. Build this
before live mic input, per the blueprint: reproducibility matters more than
demo flair at this stage, and every test in this project depends on being
able to replay the exact same "utterance" deterministically.
"""

from __future__ import annotations

import random
from typing import List

from contracts import TranscriptChunk

STEP_MS = 260.0        # midpoint of the blueprint's ~200-400ms target
JITTER_MS = 60.0
WORDS_PER_CHUNK = 2


def simulate_utterance(
    text: str,
    session_id: str,
    words_per_chunk: int = WORDS_PER_CHUNK,
    seed: int | None = 42,
    final_on_last: bool = True,
) -> List[TranscriptChunk]:
    """Deterministic by default (seed=42) so replays are reproducible for
    testing. Pass seed=None for non-deterministic jitter. Set final_on_last=False
    for trailing-off speech where no final end-of-utterance signal is emitted."""
    rng = random.Random(seed)
    words = text.strip().split()
    chunks: List[TranscriptChunk] = []
    t_ms = 0.0

    for i in range(0, len(words), words_per_chunk):
        delta = " ".join(words[i : i + words_per_chunk])
        is_final = final_on_last and ((i + words_per_chunk) >= len(words))
        t_ms += STEP_MS + rng.uniform(-JITTER_MS, JITTER_MS)
        chunks.append(
            TranscriptChunk(
                session_id=session_id,
                seq=len(chunks),
                t_ms=round(t_ms, 1),
                text_delta=delta,
                is_final=is_final,
            )
        )
    return chunks
