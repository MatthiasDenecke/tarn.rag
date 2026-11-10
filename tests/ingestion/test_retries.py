"""A failed job is retried: up to ``WORKER__MAX_ATTEMPTS`` runs in total, then dead-lettered. Both queues
mark each batch with whether it's the job's ``final_attempt``, so a failure that will be retried records
the job ``queued`` again (the document reads ``in_progress``) and only the last failure records ``failed``.

pgQueuer's default executor deletes a job whose handler raises, with no retry — the adapter registers its
``DatabaseRetryEntrypointExecutor`` and holds (not deletes) a job that exhausts its attempts."""

import asyncio
import os
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import pytest

from tarnrag.contracts import Document, PipelineItem
from tarnrag.core.engine.config import (
    DatabaseSettings,
    EmbeddingSettings,
    Settings,
    WorkerSettings,
)
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.engine.jobs import Batch, IngestionJob
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue, PgQueuerJobQueue
from tarnrag.ingestion.engine.result_sink import PassthroughSink
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.pipeline import PipelineStage


def _job(job_id: str = "j1") -> IngestionJob:
    return IngestionJob(
        job_id=job_id, document_id="d", item=PipelineItem(content="x", metadata={"source_id": "d"}),
        stage_name="Flaky", created_at=datetime.now(UTC),
    )


class FlakyStage(PipelineStage):
    """Fails its first ``failures`` calls, then succeeds — a transient error (a timeout, a lock)."""

    def __init__(self, failures: int):
        super().__init__(PipelineStage.Config(name="Flaky"))
        self.failures = failures
        self.calls = 0

    def process(self, item):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"transient #{self.calls}")
        yield item.derive()


# ---------------- the status a failed attempt records ----------------


async def test_a_retried_failure_reads_in_progress_and_only_the_last_one_failed(repo):
    orch = PipelineOrchestrator(PipelineDAG([FlakyStage(0)]), InMemoryJobQueue(), repo, {"Flaky": PassthroughSink})
    await repo.store_document(Document(content="d", metadata={"source_id": "d"}))
    ctx = await orch.begin_batch(Batch([_job()], final_attempt=False))
    await ctx.fail(RuntimeError("timeout"))
    [row] = await repo.document_jobs("d")
    assert (row["status"], row["error"]) == ("queued", "retrying after: timeout")
    assert (await repo.document_status("d"))["status"] == "in_progress"

    ctx = await orch.begin_batch(Batch([_job()], final_attempt=True))
    await ctx.fail(RuntimeError("timeout again"))
    [row] = await repo.document_jobs("d")
    assert (row["status"], row["error"]) == ("failed", "timeout again")
    assert (await repo.document_status("d"))["status"] == "failed"


# ---------------- embedded: the in-memory queue ----------------


def _wire(stage: PipelineStage, repo, **queue_options) -> tuple[PipelineOrchestrator, InMemoryJobQueue, list]:
    queue = InMemoryJobQueue(**queue_options)
    orch = PipelineOrchestrator(PipelineDAG([stage]), queue, repo, {"Flaky": PassthroughSink})
    worker = IngestionWorker(orch)
    finals: list[bool] = []

    async def handler(batch: Batch) -> None:
        finals.append(batch.final_attempt)
        await worker.handle_batch(batch)

    queue.set_handler(handler)
    return orch, queue, finals


async def test_embedded_retries_a_transient_failure_to_completion(repo):
    stage = FlakyStage(failures=2)
    orch, queue, finals = _wire(stage, repo, max_attempts=3)
    await orch.ingest_documents([PipelineItem(content="x", metadata={"source_id": "d"})])
    await queue.run()
    assert stage.calls == 3 and finals == [False, False, True]  # the third run was the last chance
    assert queue.dead_letters == []
    assert [r["status"] for r in await repo.document_jobs("d")] == ["completed"]


async def test_embedded_dead_letters_after_max_attempts(repo):
    stage = FlakyStage(failures=10)
    orch, queue, finals = _wire(stage, repo, max_attempts=2)
    await orch.ingest_documents([PipelineItem(content="x", metadata={"source_id": "d"})])
    await queue.run()
    assert stage.calls == 2 and finals == [False, True] and len(queue.dead_letters) == 1
    assert [r["status"] for r in await repo.document_jobs("d")] == ["failed"]


async def test_a_failed_multi_job_batch_is_isolated_before_any_attempt_counts(repo):
    """A batch of two where one job fails: the batch failure isn't final for anyone — each job is re-run
    alone first (isolation), and only then do attempts count."""
    stage = FlakyStage(failures=1)
    orch, queue, finals = _wire(stage, repo, max_attempts=1)
    await orch.ingest_documents([
        PipelineItem(content="x", metadata={"source_id": "d"}),
        PipelineItem(content="y", metadata={"source_id": "e"}),
    ])
    await queue.run()
    assert finals == [False, True, True]  # the 2-job batch, then each job solo on its one attempt
    assert queue.dead_letters == []


async def test_embedded_engine_takes_max_attempts_from_settings(tmp_path):
    settings = Settings(
        _env_file=None, EMBEDDING_DIMENSION=8, embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/r.db"),
        worker=WorkerSettings(max_attempts=5),
    )
    async with await IngestionEngine.create(settings) as engine:
        assert engine._queue.max_attempts == 5


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        InMemoryJobQueue(max_attempts=0)
    with pytest.raises(ValueError):
        WorkerSettings(max_attempts=0)


# ---------------- distributed: the pgQueuer adapter ----------------


def test_pgqueuer_registers_the_retrying_executor_and_holds_dead_letters():
    pytest.importorskip("pgqueuer")
    from pgqueuer.executors import DatabaseRetryEntrypointExecutor

    queue = PgQueuerJobQueue(
        driver=object(), max_attempts=4, retry_delay_seconds=2, retry_max_delay_seconds=30
    )

    async def handler(batch) -> None: ...

    queue.set_handler(handler)
    executor = queue._qm.entrypoint_registry["ingest"]
    assert isinstance(executor, DatabaseRetryEntrypointExecutor)
    assert executor.max_attempts == 3  # pgQueuer counts RE-tries: 4 runs in total
    assert (executor.initial_delay, executor.max_delay) == (timedelta(seconds=2), timedelta(seconds=30))
    assert executor.parameters.on_failure == "hold"  # a dead letter is kept, not deleted


async def test_pgqueuer_marks_the_last_attempt_final():
    pytest.importorskip("pgqueuer")
    queue = PgQueuerJobQueue(driver=object(), max_attempts=3)
    finals: list[bool] = []

    async def handler(batch: Batch) -> None:
        finals.append(batch.final_attempt)

    queue.set_handler(handler)
    run = queue._qm.entrypoint_registry["ingest"].parameters.func

    class _PgJob:
        payload = _job().model_dump_json().encode()

        def __init__(self, attempts: int) -> None:
            self.attempts = attempts

    for attempts in range(3):  # pgQueuer's attempts = the job's prior failed runs
        await run(_PgJob(attempts))
    assert finals == [False, False, True]


PG_URL = os.environ.get("TARNRAG_TEST_POSTGRES_URL", "")
requires_pg = pytest.mark.skipif(not PG_URL, reason="set TARNRAG_TEST_POSTGRES_URL to a Postgres for pgQueuer")


async def _live_queue(**options) -> PgQueuerJobQueue:
    pytest.importorskip("pgqueuer")
    pytest.importorskip("asyncpg")
    queue = await PgQueuerJobQueue.connect(
        PG_URL, dequeue_timeout_seconds=0.2, retry_delay_seconds=0.05, retry_max_delay_seconds=0.1, **options
    )
    with suppress(Exception):
        await queue._queries.uninstall()
    await queue._queries.install()
    return queue


async def _run_until(queue: PgQueuerJobQueue, done: asyncio.Event, timeout: float = 20) -> None:
    task = asyncio.create_task(queue.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
        await asyncio.sleep(0.3)  # let pgQueuer write the outcome of the last run
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.requires_postgres
@requires_pg
async def test_pgqueuer_retries_a_failed_job_until_it_succeeds():
    queue = await _live_queue(max_attempts=3)
    finals: list[bool] = []
    done = asyncio.Event()

    async def handler(batch: Batch) -> None:
        finals.append(batch.final_attempt)
        if len(finals) == 1:
            raise RuntimeError("transient")
        done.set()

    try:
        await queue.enqueue(_job())
        queue.set_handler(handler)
        await _run_until(queue, done)
        assert finals == [False, False]  # failed once, re-queued by pgQueuer, succeeded on run two
        assert await queue._queries.list_failed_jobs() == []
    finally:
        with suppress(Exception):
            await queue._queries.uninstall()
        await queue.aclose()


@pytest.mark.requires_postgres
@requires_pg
async def test_pgqueuer_holds_a_job_that_exhausts_its_attempts():
    queue = await _live_queue(max_attempts=2)
    finals: list[bool] = []
    done = asyncio.Event()

    async def handler(batch: Batch) -> None:
        finals.append(batch.final_attempt)
        if batch.final_attempt:
            done.set()
        raise RuntimeError("permanent")

    try:
        await queue.enqueue(_job())
        queue.set_handler(handler)
        await _run_until(queue, done)
        assert finals == [False, True]  # two runs in total, then no more
        [held] = await queue._queries.list_failed_jobs()  # the dead letter is kept for inspection
        assert IngestionJob.model_validate_json(held.payload).job_id == "j1"
    finally:
        with suppress(Exception):
            await queue._queries.uninstall()
        await queue.aclose()
