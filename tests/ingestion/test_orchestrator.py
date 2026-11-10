import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.core.exceptions import IngestionError
from tarnrag.ingestion.engine.jobs import Batch
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import JobEnqueuer
from tarnrag.ingestion.engine.result_sink import (
    FinalizationOutcome,
    PassthroughSink,
    ResultSink,
    create_sink_registry,
)
from tarnrag.ingestion.pipeline.pipeline import MapperStage


class RecordingEnqueuer(JobEnqueuer):
    """A JobEnqueuer that just records enqueued jobs."""

    def __init__(self):
        self.enqueued: list = []

    async def enqueue(self, job) -> None:
        self.enqueued.append(job)


class _NoopStage(MapperStage):
    def __init__(self, name: str):
        super().__init__(MapperStage.Config(name=name))

    def map(self, text, metadata):
        return text, {}


def _dag(*names):
    return PipelineDAG([_NoopStage(n) for n in names])


def _item(source_id="s1", **meta):
    return PipelineItem(content="x", metadata={"source_id": source_id, **meta})


# ---------------- DAG naming (A3: names must be deterministic across processes) ----------------


def test_stage_names_are_deterministic_across_builds():
    """A3: node names derive from the pipeline definition alone. A producer enqueues jobs whose
    ``stage_name`` must resolve in the CONSUMER's DAG — a different process with its own construction
    history — so two builds of the same Settings must agree, whatever was constructed in between."""
    from tarnrag.core.engine.config import Settings
    from tarnrag.ingestion.engine.engine import IngestionEngine
    from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage

    settings = Settings(_env_file=None)
    producer = PipelineDAG(IngestionEngine.build_pipeline(settings).stages)
    CleanAndNormalizeStage(CleanAndNormalizeStage.Config())  # unrelated construction between builds
    consumer = PipelineDAG(IngestionEngine.build_pipeline(settings).stages)

    assert [s.name for s in producer.stages] == [s.name for s in consumer.stages]
    for job_stage_name in (s.name for s in producer.stages):  # every producer job routes
        assert consumer.get_stage(job_stage_name) is not None


def test_dag_names_unnamed_stages_by_position():
    """Unnamed stages get ``{tag}-{position}`` — two same-class stages coexist with distinct,
    deterministic identities."""
    from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage

    dag = PipelineDAG([CleanAndNormalizeStage(CleanAndNormalizeStage.Config()) for _ in range(2)])
    assert [s.name for s in dag.stages] == ["CleanAndNormalize-0", "CleanAndNormalize-1"]


def test_explicit_stage_names_are_kept():
    dag = _dag("first", "second")
    assert [s.name for s in dag.stages] == ["first", "second"]


def test_dag_refuses_duplicate_stage_names():
    """Two nodes with one name would silently route every job to the first match — fail at build."""
    with pytest.raises(ValueError, match="duplicate stage name"):
        _dag("same", "same")


async def test_ingest_documents_enqueues_root_jobs(repo):
    q = RecordingEnqueuer()
    orch = PipelineOrchestrator(
        _dag("LoadAndParse", "CleanAndNormalize"), q, repo, create_sink_registry()
    )
    ids = await orch.ingest_documents([_item("s1")])
    assert ids == ["s1"]
    assert len(q.enqueued) == 1
    assert q.enqueued[0].stage_name == "LoadAndParse"
    assert q.enqueued[0].document_id == "s1"
    assert (await repo.document_status("s1"))["status"] == "pending"  # queued, no data yet


async def test_complete_fans_out_one_job_per_item(repo):
    q = RecordingEnqueuer()
    # CleanAndNormalize -> Chunk; Clean's sink is PassthroughSink (persists nothing).
    orch = PipelineOrchestrator(_dag("CleanAndNormalize", "Chunk"), q, repo, create_sink_registry())
    job = orch._make_job(_item("s1"), "CleanAndNormalize")
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([_item("s1") for _ in range(3)])  # 3 produced -> 3 downstream Chunk jobs
    await ctx.complete()
    assert len(q.enqueued) == 3
    assert all(j.stage_name == "Chunk" for j in q.enqueued)


async def test_complete_terminal_enqueues_nothing(repo):
    q = RecordingEnqueuer()
    orch = PipelineOrchestrator(_dag("CleanAndNormalize", "Embed"), q, repo, create_sink_registry())
    job = orch._make_job(_item("s1"), "Embed")  # Embed is terminal
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([])  # terminal: no embeddings to persist here, no downstream regardless
    await ctx.complete()
    assert q.enqueued == []


async def test_complete_persistence_failure_raises_and_records(repo):
    q = RecordingEnqueuer()

    class FailingSink(ResultSink):
        def __init__(self, repository):
            pass

        def submit(self, results):
            pass

        def close(self):
            pass

        async def finalize(self):
            return FinalizationOutcome(persisted=False, detail="disk on fire")

    orch = PipelineOrchestrator(
        _dag("X", "Y"), q, repo, {"X": FailingSink, "Y": PassthroughSink}
    )
    job = orch._make_job(_item("s1"), "X")
    ctx = await orch.begin_batch(Batch([job]))
    ctx.submit([_item("s1")])
    with pytest.raises(IngestionError):
        await ctx.complete()
    assert q.enqueued == []  # no downstream on failure
    assert (await repo.document_status("s1"))["status"] == "failed"


def test_dag_edges_and_terminal():
    dag = _dag("A", "B", "C")
    assert dag.get_downstream_stages("A") == ["B"]
    assert dag.get_downstream_stages("C") == []  # terminal
