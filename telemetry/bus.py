"""
M6 — Telemetry bus.

"Non-blocking" in the blueprint means the rest of the pipeline should never
wait on telemetry I/O. A single local JSONL file append is fast enough that
a synchronous write doesn't meaningfully block your latency budget at
hackathon scale — if you later see telemetry writes on your latency
profile's hot path (Phase 3, Step 9), swap `emit()`'s body for an
asyncio.Queue drained by a background task instead. Don't build that
upfront; it's premature complexity this stage doesn't need yet.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import List

from contracts import TelemetryEvent


class TelemetryBus:
    def __init__(self, jsonl_path: str = "telemetry/trace_log.jsonl") -> None:
        self._path = Path(jsonl_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: TelemetryEvent) -> None:
        line = json.dumps(asdict(event), default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def events_for_trace(self, trace_id: str) -> List[dict]:
        """Read back every event for one trace_id — this is how you verify
        G6 (100% trace coverage) for a given turn, and how the demo video's
        'telemetry showing 0 queries' shot gets its data."""
        if not self._path.exists():
            return []
        out = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("trace_id") == trace_id:
                    out.append(ev)
        return out

    def stage_coverage(self, trace_id: str) -> List[str]:
        """Convenience for G6 checks: which stages fired for this trace."""
        return [ev["stage"] for ev in self.events_for_trace(trace_id)]
