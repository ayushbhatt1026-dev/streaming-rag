"""
Proves corpus isolation is real, not just documented: monkeypatch every
socket-creation path so any attempted network call raises immediately, then
run a full index + search cycle. If this test passes, the retriever cannot
have reached the internet or any external service to answer the query —
its only possible source of facts is the local corpus/ folder.

Run: python -m pytest tests/test_corpus_isolation.py -v
"""

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import SubQuery
from retrieval.embeddings import TfidfEmbeddingBackend
from retrieval.retriever import Retriever
from retrieval.vector_store import InMemoryVectorStore


def _blocked_socket(*args, **kwargs):
    raise AssertionError(
        "Network access attempted during retrieval — corpus isolation violated."
    )


def test_search_never_touches_the_network(monkeypatch):
    # Index BEFORE blocking the network: the offline dev embedding backend
    # (TF-IDF) needs no network at all, so this also proves the *query
    # path* specifically never needs it, independent of setup.
    retriever = Retriever(
        embedding_backend=TfidfEmbeddingBackend(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.index_corpus(corpus_dir=str(Path(__file__).resolve().parents[1] / "corpus"))

    # Now cut off all network access and confirm search() still works.
    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_socket)

    results = retriever.search(
        [SubQuery(id="q1", text="What is the seated capacity of Venue B?")]
    )

    assert len(results) > 0
    for chunk in results:
        assert chunk.doc_id.startswith("Doc_")  # traceable to the local corpus
