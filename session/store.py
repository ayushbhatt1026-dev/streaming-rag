"""
M4 — SessionStore.

Session-Bound State is a hard rule (guide §3), not just good practice:
in-memory only, TTL-expiring, no persistence layer, no cross-session user
profile. `reset_session` must exist and be demonstrably used between
benchmark runs — the guide calls this out explicitly as something teams
lose points for never actually testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from contracts import AnswerVersion


@dataclass
class _SessionRecord:
    answer_version: AnswerVersion
    updated_at: float


class SessionStore:
    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._sessions: Dict[str, _SessionRecord] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> Optional[AnswerVersion]:
        self._expire()
        record = self._sessions.get(session_id)
        return record.answer_version if record else None

    def set(self, session_id: str, answer_version: AnswerVersion) -> None:
        self._sessions[session_id] = _SessionRecord(answer_version, time.time())

    def reset_session(self, session_id: str) -> None:
        """Explicit teardown. Call this between benchmark runs and in your
        eval harness setup/teardown — the guide requires this endpoint to
        exist AND be demonstrably exercised, not just defined."""
        self._sessions.pop(session_id, None)

    def active_session_ids(self) -> list[str]:
        """For debugging/telemetry only — never used to build a
        cross-session profile."""
        self._expire()
        return list(self._sessions.keys())

    def _expire(self) -> None:
        now = time.time()
        expired = [
            sid for sid, rec in self._sessions.items()
            if now - rec.updated_at > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
