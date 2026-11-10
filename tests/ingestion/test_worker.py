from datetime import UTC, datetime

import pytest

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.engine.batch import BatchContext, BatchCoordinator
from tarnrag.ingestion.engine.jobs import IngestionJob
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue
from tarnrag.ingestion.engine.result_sink import PassthroughSink
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.pipeline import PipelineStage


class BoomStage(PipelineStage):
    """A stage that always fails during compute."""

    def __init__(self):
        super().__init__(PipelineStage.Config(name="Boom"))

    def process(self, item):
        raise RuntimeError("boom")
        yield  # pragma: no cover - marks this a generator


def _wire(queue, repo):
    dag = PipelineDAG([BoomStage()])
    orch = PipelineOrchestrator(dag, queue, repo, {"Boom": PassthroughSink})
    worker = IngestionWorker(orch)  # pure handler
    queue.set_handler(worker.handle_batch)  # composition-root wiring
    return orch


async def test_compute_error_marks_failed_and_propagates(repo):
    queue = InMemoryJobQueue(requeue_on_error=False)  # raise-through
    orch = _wire(queue, repo)
    await orch.ingest_documents([PipelineItem(content="x", metadata={"source_id": "s1"})])
    with pytest.raises(RuntimeError, match="boom"):
        await queue.run()
    jobs = await repo.document_jobs("s1")
    assert any(j["status"] == "failed" and j["error"] == "boom" for j in jobs)


async def test_recovery_requeues_then_dead_letters(repo):
    queue = InMemoryJobQueue(max_attempts=2)  # retry once, then dead-letter
    orch = _wire(queue, repo)
    await orch.ingest_documents([PipelineItem(content="x", metadata={"source_id": "s1"})])
    await queue.run()  # exhausts retries without raising
    assert len(queue.dead_letters) == 1
    assert (await repo.document_status("s1"))["status"] == "failed"


async def test_worker_is_pure_handler(repo):
    # The worker holds no queue and exposes no run loop — just handle_batch.
    orch = PipelineOrchestrator(
        PipelineDAG([BoomStage()]), InMemoryJobQueue(), repo, {"Boom": PassthroughSink}
    )
    worker = IngestionWorker(orch)
    assert callable(worker.handle_batch)
    assert not hasattr(worker, "queue")
    assert not hasattr(worker, "run")


async def test_worker_reports_to_context_without_orchestrator():
    """Drive the worker with a fake coordinator/context — no orchestrator, repo, or
    real queue logic — proving the worker is decoupled from the orchestrator."""
    events: list = []

    class FakeContext(BatchContext):
        def submit(self, results):
            events.append(("submit", list(results)))

        async def fail(self, error):
            events.append(("fail", str(error)))

        async def complete(self):
            events.append(("complete",))

    class EchoStage(PipelineStage):
        def __init__(self):
            super().__init__(PipelineStage.Config(name="Echo"))

        def process(self, item):
            yield item

    class FakeCoordinator(BatchCoordinator):
        def get_stage(self, stage_name):
            events.append(("get_stage", stage_name))
            return EchoStage()

        async def begin_batch(self, batch):
            events.append(("begin", batch.stage_name))
            return FakeContext()

    queue = InMemoryJobQueue()
    worker = IngestionWorker(FakeCoordinator())
    queue.set_handler(worker.handle_batch)
    await queue.enqueue(
        IngestionJob(
            job_id="j1",
            document_id="s1",
            item=PipelineItem(content="x", metadata={"source_id": "s1"}),
            stage_name="Echo",
            created_at=datetime.now(UTC),
        )
    )
    await queue.run()

    assert ("begin", "Echo") in events
    assert ("get_stage", "Echo") in events
    assert any(e[0] == "submit" for e in events)
    assert ("complete",) in events
    assert not any(e[0] == "fail" for e in events)
