"""
M3 — vector store backends.

Decision log choice: Qdrant.

Note on "local": Qdrant does not require Docker or a server to be a "real
local search engine." `qdrant-client` ships an embedded mode that persists
to a folder on disk (RocksDB under the hood) with the exact same API as
talking to a Qdrant server — this is the simplest way to satisfy "local
search engine" for a hackathon, and you can point the same code at a
docker-compose Qdrant server later by changing one constructor argument
if you want the dashboard UI.

Both backends implement the same tiny interface, so retriever.py doesn't
care which one is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np


class VectorStore(Protocol):
    def upsert(self, ids: List[str], vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Return [(id, cosine_score), ...] sorted descending, best first."""
        ...


class QdrantVectorStore:
    """Production backend. Requires: pip install qdrant-client

    Runs fully local via embedded (on-disk) mode — no Docker, no server,
    no network access needed at query time. If you later want the Qdrant
    web dashboard for debugging, swap `path=` for `url="http://localhost:6333"`
    against a `docker compose up` Qdrant service; nothing else changes.
    """

    def __init__(self, collection_name: str = "corpus_chunks", dim: int = 384,
                 path: str = "./qdrant_data") -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. Run:\n  pip install qdrant-client"
            ) from e

        self._PointStruct = PointStruct
        self._client = QdrantClient(path=path)
        self._collection = collection_name

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._id_map: dict[int, str] = {}
        self._next_int_id = 0

    def upsert(self, ids: List[str], vectors: np.ndarray) -> None:
        points = []
        for chunk_id, vec in zip(ids, vectors):
            int_id = self._next_int_id
            self._id_map[int_id] = chunk_id
            points.append(self._PointStruct(id=int_id, vector=vec.tolist()))
            self._next_int_id += 1
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        # qdrant-client >=1.10 deprecated (and later versions removed)
        # .search() in favor of .query_points(). Try the current API first,
        # fall back to the old one so this still works on older installs.
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector.tolist(),
                limit=top_k,
            )
            hits = response.points
        except AttributeError:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
            )
        return [(self._id_map[h.id], float(h.score)) for h in hits]


@dataclass
class InMemoryVectorStore:
    """OFFLINE DEV FALLBACK. Plain numpy cosine similarity — correct, just
    not indexed for scale. Fine for a hackathon-sized corpus (hundreds to a
    few thousand chunks). Swap to QdrantVectorStore before you report
    latency numbers, since this does a full linear scan."""

    def __post_init__(self) -> None:
        self._ids: List[str] = []
        self._vectors: np.ndarray | None = None

    def upsert(self, ids: List[str], vectors: np.ndarray) -> None:
        self._ids = list(ids)
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        sims = self._vectors @ q  # vectors are pre-normalized -> cosine sim
        top_idx = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]
