"""HIGH-08: ``settings.worker`` (``WORKER__CONCURRENCY`` / ``WORKER__QUEUE_TIMEOUT_SECONDS``) reaches the
pgQueuer consumer loop — it used to be read nowhere."""

from datetime import timedelta

import pytest

from tarnrag.core.engine.config import (
    DatabaseSettings,
    EmbeddingSettings,
    Settings,
    WorkerSettings,
)
from tarnrag.core.resources.embedder import Embedder
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.engine.queue import PgQueuerJobQueue


@pytest.mark.parametrize(("concurrency", "batch", "claimed"), [(1, 1, 2), (2, 1, 2), (4, 2, 4), (5, 2, 5)])
async def test_the_run_loop_applies_concurrency_and_dequeue_timeout(monkeypatch, concurrency, batch, claimed):
    """pgQueuer insists on ``max_concurrent_tasks >= 2 * batch_size`` (it raises otherwise — a bare
    ``max_concurrent_tasks=concurrency`` would kill the loop); the derived pair always satisfies it."""
    pytest.importorskip("pgqueuer")
    queue = PgQueuerJobQueue(driver=object(), concurrency=concurrency, dequeue_timeout_seconds=7)
    seen: dict = {}

    async def _run(**kwargs) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(queue._qm, "run", _run)
    await queue.run()
    assert seen == {
        "dequeue_timeout": timedelta(seconds=7), "batch_size": batch, "max_concurrent_tasks": claimed,
    }
    assert seen["max_concurrent_tasks"] >= 2 * seen["batch_size"]


async def test_in_flight_batches_are_capped_at_the_configured_concurrency():
    """The execution cap is exact — even at concurrency=1, below pgQueuer's own floor of 2."""
    import asyncio

    pytest.importorskip("pgqueuer")
    queue = PgQueuerJobQueue(driver=object(), concurrency=1)
    running = peak = 0

    async def handler(batch) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    queue.set_handler(handler)
    entry = queue._qm.entrypoint_registry["ingest"].parameters.func  # the registered coroutine

    class _PgJob:
        payload = _job_payload()
        attempts = 0

    await asyncio.gather(*(entry(_PgJob()) for _ in range(3)))
    assert peak == 1


def _job_payload() -> bytes:
    from datetime import UTC, datetime

    from tarnrag.contracts import PipelineItem
    from tarnrag.ingestion.engine.jobs import IngestionJob

    return IngestionJob(
        job_id="j", document_id="d", item=PipelineItem(content="x", metadata={}),
        stage_name="Chunk", created_at=datetime.now(UTC),
    ).model_dump_json().encode()


def test_concurrency_must_be_positive():
    pytest.importorskip("pgqueuer")
    with pytest.raises(ValueError, match="concurrency"):
        PgQueuerJobQueue(driver=object(), concurrency=0)
    with pytest.raises(ValueError):
        WorkerSettings(concurrency=0)


async def test_distributed_engine_builds_its_queue_from_worker_settings(monkeypatch, repo):
    captured: dict = {}

    class _Queue:
        async def enqueue(self, job) -> None: ...

    async def _connect(url: str, **kwargs):
        captured.update(url=url, **kwargs)
        return _Queue()

    monkeypatch.setattr(PgQueuerJobQueue, "connect", staticmethod(_connect))
    settings = Settings(
        _env_file=None, MODE="distributed", EMBEDDING_DIMENSION=3,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url="postgresql://u@h/docs", queue_url="postgresql://u@h/queue"),
        worker=WorkerSettings(
            concurrency=2, queue_timeout_seconds=12, max_attempts=5, retry_delay_seconds=0.5,
            retry_max_delay_seconds=60,
        ),
    )
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    await IngestionEngine.create(settings, repository=repo, embedder=embedder)  # the store is injected
    assert captured == {
        "url": "postgresql://u@h/queue", "concurrency": 2, "dequeue_timeout_seconds": 12,
        "max_attempts": 5, "retry_delay_seconds": 0.5, "retry_max_delay_seconds": 60,
    }
