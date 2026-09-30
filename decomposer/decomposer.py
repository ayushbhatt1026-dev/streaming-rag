"""
M2 — MultiIntentDecomposer. This is what the Controller (M1) calls after a
RETRIEVE_* decision. Always run the anti-fragmentation guard, even on a
single proposed sub-query — it's a no-op there, but keeps the code path
uniform regardless of backend.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import yaml

from contracts import SubQuery, TelemetryEvent
from decomposer.anti_fragmentation import cap_by_distinctness, merge_near_duplicates
from decomposer.backends import DecomposeBackend
from retrieval.embeddings import EmbeddingBackend
from telemetry.bus import TelemetryBus


def _load_config(path: str = "config/decomposer.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class MultiIntentDecomposer:
    backend: DecomposeBackend
    embedding_backend: EmbeddingBackend
    telemetry: TelemetryBus
    session_id: str
    trace_id: str
    config_path: str = "config/decomposer.yaml"

    def __post_init__(self) -> None:
        cfg = _load_config(self.config_path)
        self._merge_threshold = cfg["merge_similarity_threshold"]
        self._max_sub_queries = cfg["max_sub_queries"]

    def decompose(self, utterance: str, parent_utterance_id: str) -> list[SubQuery]:
        start = time.perf_counter()

        raw = self.backend.propose(utterance)
        merged = merge_near_duplicates(raw, self.embedding_backend, self._merge_threshold)
        capped = cap_by_distinctness(merged, self.embedding_backend, self._max_sub_queries)

        sub_queries = [
            SubQuery(
                id=str(uuid.uuid4())[:8],
                text=c["text"],
                intent_type=c.get("intent_type", "factual_lookup"),
                parent_utterance_id=parent_utterance_id,
            )
            for c in capped
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        self.telemetry.emit(
            TelemetryEvent(
                trace_id=self.trace_id,
                session_id=self.session_id,
                stage="decomposer",
                event="decomposition_completed",
                t_ms=0.0,
                payload={
                    "n_raw": len(raw),
                    "n_after_merge": len(merged),
                    "n_final": len(capped),
                    "latency_ms": round(latency_ms, 3),
                    "sub_queries": [sq.text for sq in sub_queries],
                },
            )
        )
        return sub_queries
