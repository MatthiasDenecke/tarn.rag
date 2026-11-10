"""Observability is actually emitted by the worker (compute) and orchestrator (lifecycle).

Drives the real InMemory + SQLite flow with a recording Observability on both, and asserts
the expected metrics/logs appear. Stages remain pure — they get no obs.
"""

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.core.engine.observability import Observability
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue
from tarnrag.ingestion.engine.result_sink import PassthroughSink, create_sink_registry
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.embed import EmbedStage
from tarnrag.ingestion.pipeline.pipeline import PipelineStage


class _RecordingObs(Observability):
    def __init__(self):
        self.counters: list[tuple[str, int]] = []
        self.gauges: list[str] = []
        self.logs: list[tuple[str, str]] = []

    async def log(self, level, message, **context):
        self.logs.append((level, message))

    def counter(self, name, value=1, tags=None):
        self.counters.append((name, value))

    def gauge(self, name, value, tags=None):
        self.gauges.append(name)

    @property
    def counter_names(self) -> set[str]:
        return {n for n, _ in self.counters}


class _FakeEmbedder:
    def embed_passages(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def _embed_stage():
    """A real EmbedStage with the fake encoder injected (no model needed)."""
    stage = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(batch_size=2)))
    stage._embedder = _FakeEmbedder()
    return stage


def _stages():
    return [
        LoadAndParseStage(LoadAndParseStage.Config()),
        CleanAndNormalizeStage(CleanAndNormalizeStage.Config()),
        ChunkStage(ChunkStage.Config(chunker={"class_name": "recursive", "chunk_size": 30, "overlap": 5})),
        _embed_stage(),
    ]


async def test_metrics_emitted_on_successful_ingest(repo):
    obs = _RecordingObs()
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(
        PipelineDAG(_stages()), queue, repo, create_sink_registry(), observability=obs
    )
    worker = IngestionWorker(orch, observability=obs)
    queue.set_handler(worker.handle_batch)

    await orch.ingest_documents(
        [PipelineItem(content="Hello world. " * 10, metadata={"source_id": "s1"})]
    )
    await queue.run()

    # Orchestrator lifecycle metrics.
    assert ("ingest.documents", 1) in obs.counters
    assert "ingest.jobs_enqueued" in obs.counter_names
    assert any(n.endswith(".completed") for n in obs.counter_names)
    # Worker compute metrics: per-stage timer gauge + throughput counter.
    assert any(g.endswith(".process.seconds") for g in obs.gauges)
    assert "stage.Embed.items" in obs.counter_names


class BoomStage(PipelineStage):
    def __init__(self):
        super().__init__(PipelineStage.Config(name="Boom"))

    def process(self, item):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    def validate(self) -> None:
        return None


async def test_compute_failure_emits_error_counter_and_log(repo):
    obs = _RecordingObs()
    queue = InMemoryJobQueue(requeue_on_error=False)  # raise-through
    orch = PipelineOrchestrator(
        PipelineDAG([BoomStage()]), queue, repo, {"Boom": PassthroughSink}, observability=obs
    )
    worker = IngestionWorker(orch, observability=obs)
    queue.set_handler(worker.handle_batch)

    await orch.ingest_documents(
        [PipelineItem(content="x", metadata={"source_id": "s1"})]
    )
    with pytest.raises(RuntimeError, match="boom"):
        await queue.run()

    assert "stage.Boom.errors" in obs.counter_names
    assert any(level == "error" for level, _ in obs.logs)
