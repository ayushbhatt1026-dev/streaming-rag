"""
The "empty app service" from Phase 0, Step 3.

This is intentionally tiny — its only job right now is to prove the whole
docker compose stack (app + Qdrant) can start and talk to each other. Real
endpoints (the WebSocket transcript ingestion, etc.) get added starting in
Phase 1's walking skeleton — don't add them yet.
"""

import os

from fastapi import FastAPI

app = FastAPI(title="Streaming Live RAG")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


@app.get("/health")
def health():
    return {"status": "ok", "qdrant_url": QDRANT_URL}
