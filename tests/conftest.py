"""
Shared pytest fixtures for the Step 4-7 test suites.

Everything here runs OFFLINE and deterministically:
  - TF-IDF embeddings (no BGE download)
  - InMemoryVectorStore (no Qdrant lock / no qdrant_data folder touched)
  - Heuristic Stage-2 / decomposer backends (no Gemini key, no quota burn)
  - a fresh telemetry JSONL per test (tmp_path), never telemetry/trace_log.jsonl

Run everything:  python -m pytest tests -v
"""

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from controller.controller import RetrievalController  # noqa: E402
from controller.stage2 import HeuristicStage2Confirm  # noqa: E402
from decomposer.backends import HeuristicDecomposeBackend  # noqa: E402
from decomposer.decomposer import MultiIntentDecomposer  # noqa: E402
from orchestrator.walking_skeleton import run_one_utterance  # noqa: E402
from retrieval.embeddings import TfidfEmbeddingBackend  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from retrieval.vector_store import InMemoryVectorStore  # noqa: E402
from synthesis.naive_synthesizer import NaiveSynthesizer  # noqa: E402
from telemetry.bus import TelemetryBus  # noqa: E402

CORPUS_DIR = str(ROOT / "corpus")
CONTROLLER_CFG = str(ROOT / "config" / "controller.yaml")
DECOMPOSER_CFG = str(ROOT / "config" / "decomposer.yaml")


@pytest.fixture(scope="session")
def embedding_backend():
    return TfidfEmbeddingBackend()


@pytest.fixture(scope="session")
def retriever(embedding_backend):
    r = Retriever(embedding_backend=embedding_backend, vector_store=InMemoryVectorStore())
    r.index_corpus(corpus_dir=CORPUS_DIR)  # also fits the TF-IDF backend
    return r


@pytest.fixture
def telemetry(tmp_path):
    return TelemetryBus(jsonl_path=str(tmp_path / "trace.jsonl"))


@pytest.fixture
def make_controller(telemetry):
    def _make(stage2=None, config_path=CONTROLLER_CFG, session_id="sess-test", trace_id="trace-test", **kw):
        return RetrievalController(
            stage2=stage2 or HeuristicStage2Confirm(),
            telemetry=telemetry,
            session_id=session_id,
            trace_id=trace_id,
            config_path=config_path,
            **kw,
        )
    return _make


@pytest.fixture
def make_decomposer(telemetry, embedding_backend, retriever):  # retriever fixture => TF-IDF is fitted
    def _make(backend=None, config_path=DECOMPOSER_CFG, session_id="sess-test", trace_id="trace-test"):
        return MultiIntentDecomposer(
            backend=backend or HeuristicDecomposeBackend(),
            embedding_backend=embedding_backend,
            telemetry=telemetry,
            session_id=session_id,
            trace_id=trace_id,
            config_path=config_path,
        )
    return _make


@dataclass
class Turn:
    result: object
    telemetry: TelemetryBus
    trace_id: str

    def events(self):
        return self.telemetry.events_for_trace(self.trace_id)

    def events_named(self, name):
        return [e for e in self.events() if e["event"] == name]


@pytest.fixture
def run_turn(telemetry, embedding_backend, retriever):
    """Run one utterance through the REAL skeleton (Steps 4-6 wiring)."""
    def _run(utterance, stage2=None, backend=None, session_id="sess-test"):
        trace_id = str(uuid.uuid4())
        controller = RetrievalController(
            stage2=stage2 or HeuristicStage2Confirm(), telemetry=telemetry,
            session_id=session_id, trace_id=trace_id, config_path=CONTROLLER_CFG,
        )
        decomposer = MultiIntentDecomposer(
            backend=backend or HeuristicDecomposeBackend(), embedding_backend=embedding_backend,
            telemetry=telemetry, session_id=session_id, trace_id=trace_id,
            config_path=DECOMPOSER_CFG,
        )
        result = run_one_utterance(
            utterance, controller, decomposer, retriever, NaiveSynthesizer(),
            telemetry, session_id, trace_id,
        )
        return Turn(result=result, telemetry=telemetry, trace_id=trace_id)
    return _run


class FakeLLMClient:
    """Stands in for GeminiLLMClient (same `complete_json` surface)."""

    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.exc is not None:
            raise self.exc
        return self.response
