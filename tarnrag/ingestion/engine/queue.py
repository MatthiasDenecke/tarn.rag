"""Job queue ports + adapters.

The queue surface is segregated by role: ``JobEnqueuer`` (producer — the orchestrator
puts jobs on the queue) and ``JobConsumer`` (the runtime — register a handler and run
the dispatch loop, which belongs to the queue technology). One adapter implements both:
``PgQueuerJobQueue`` (the only place that touches pgQueuer, lazily imported) and
``InMemoryJobQueue`` (a test double that runs the whole flow with no Postgres/pgQueuer).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from tarnrag.ingestion.engine.jobs import Batch, IngestionJob

ENTRYPOINT = "ingest"  # single pgQueuer entrypoint; the consumer forms homogeneous Batches

# A handler receives a Batch — a homogeneous unit (all jobs share one stage_name; the Batch
# constructor enforces it). Forming the Batch is the consumer's job; the worker just runs it.
# The InMemory queue groups each wave into real multi-job batches per stage; the pgQueuer
# adapter still dispatches single-job batches (claim-side batching is a pending optimization).
JobHandler = Callable[[Batch], Awaitable[None]]


class JobEnqueuer(ABC):
    """
    Producer side — put jobs on the queue. Used by the orchestrator.
    """

    @abstractmethod
    async def enqueue(self, job: IngestionJob) -> None: ...


class JobConsumer(ABC):
    """
    Consumer/runtime side — register a handler and run the dispatch loop. The loop
    itself belongs to the queue technology (e.g. pgQueuer's QueueManager); the worker
    is just a handler, and the composition root calls ``set_handler`` then ``run``.
    Mechanics (claiming, retries, NOTIFY, dead-lettering) belong to the implementation.
    """

    @abstractmethod
    def set_handler(self, handler: JobHandler) -> None: ...

    @abstractmethod
    async def run(self) -> None: ...


class InMemoryJobQueue(JobEnqueuer, JobConsumer):
    """
    The **production embedded-mode queue** (and the test double) — in-process, no Postgres, no
    pgQueuer. Emulates pgQueuer's at-least-once + requeue-on-raise semantics so embedded mode and
    the tests behave like the distributed queue.

    ``requeue_on_error=True`` (default) re-queues a failing job up to ``max_attempts``
    (recovery tests); ``False`` re-raises immediately (sharp unit failures). pgQueuer
    does NOT guarantee ordering — tests must not rely on it.
    """

    def __init__(self, requeue_on_error: bool = True, max_attempts: int = 3, max_batch_size: int = 256):
        self._jobs: asyncio.Queue[tuple[IngestionJob, int]] = asyncio.Queue()
        self._handler: JobHandler | None = None
        self.requeue_on_error = requeue_on_error
        self.max_attempts = max_attempts
        self.max_batch_size = max_batch_size  # jobs grouped per dispatch (one stage runs over many docs at once)
        self.dead_letters: list[IngestionJob] = []

    async def enqueue(self, job: IngestionJob) -> None:
        await self._jobs.put((job, 0))

    def set_handler(self, handler: JobHandler) -> None:
        self._handler = handler

    async def run(self) -> None:
        """Drain all jobs (including any enqueued while handling) and stop — so tests terminate. Each wave
        drains the currently-queued jobs and **dispatches them in homogeneous batches** (grouped by
        ``stage_name``, capped at ``max_batch_size``), so a stage runs over many documents at once — Embed
        batches their chunks, and each sink commits once per batch instead of once per document. The DAG
        advance re-enqueues next-stage jobs, which the next wave picks up."""
        assert self._handler is not None, "set_handler() before run()"
        while not self._jobs.empty():
            wave: list[tuple[IngestionJob, int]] = []
            while not self._jobs.empty():
                wave.append(self._jobs.get_nowait())  # snapshot the current queue (this stage-wave)
            by_stage: dict[str, list[tuple[IngestionJob, int]]] = {}
            for entry in wave:
                by_stage.setdefault(entry[0].stage_name, []).append(entry)
            for stage_jobs in by_stage.values():
                for i in range(0, len(stage_jobs), self.max_batch_size):
                    await self._dispatch(stage_jobs[i : i + self.max_batch_size])

    async def _dispatch(self, batch: list[tuple[IngestionJob, int]]) -> None:
        """Dispatch one homogeneous batch. On failure with requeue, **isolate**: re-run each job solo so a
        single bad job doesn't penalize the rest — preserving the per-job at-least-once / dead-letter
        semantics (a solo failure requeues that job or, past ``max_attempts``, dead-letters it)."""
        try:
            await self._handler(Batch([job for job, _ in batch]))
        except Exception:
            if not self.requeue_on_error:
                raise
            if len(batch) == 1:
                job, attempts = batch[0]
                if attempts + 1 >= self.max_attempts:
                    self.dead_letters.append(job)
                else:
                    await self._jobs.put((job, attempts + 1))
            else:  # a batched failure — re-run each job alone to find (and isolate) the culprit
                for entry in batch:
                    await self._dispatch([entry])


class PgQueuerJobQueue(JobEnqueuer, JobConsumer):
    """
    Queue backed by pgQueuer — the only adapter that imports pgQueuer (lazily).
    pgQueuer owns claiming/retries/NOTIFY/dead-lettering; a handler that raises
    propagates so pgQueuer requeues the job (= recovery, D5).
    """

    def __init__(self, driver, pool=None):
        from pgqueuer import QueueManager
        from pgqueuer.queries import Queries

        # pgQueuer's QueueManager takes a Queries (it reads its query builders, e.g. queries.qbe), not a
        # raw driver — passing the driver makes run()/verify_structure fail with AttributeError.
        self._queries = Queries(driver)
        self._qm = QueueManager(self._queries)
        self._pool = pool  # the asyncpg pool connect() opened (None when a driver is injected)

    @classmethod
    async def connect(cls, connection_url: str) -> PgQueuerJobQueue:
        """
        Create an asyncpg pool for ``connection_url`` and wrap it in a pgQueuer driver.
        """
        import asyncpg
        from pgqueuer.db import AsyncpgPoolDriver

        pool = await asyncpg.create_pool(connection_url)
        return cls(AsyncpgPoolDriver(pool), pool=pool)

    async def aclose(self) -> None:
        """Release the asyncpg pool :meth:`connect` opened. Idempotent; a driver-injected instance owns
        no pool and closes nothing (its owner does)."""
        if self._pool is not None:
            pool, self._pool = self._pool, None
            await pool.close()

    async def enqueue(self, job: IngestionJob) -> None:
        # payload is the full IngestionJob (incl. the inline item, D1)
        await self._queries.enqueue([ENTRYPOINT], [job.model_dump_json().encode()], [0])

    def set_handler(self, handler: JobHandler) -> None:
        """
        Register ``handler`` as the pgQueuer entrypoint; each claimed job is decoded into a
        single-job ``Batch`` and handed over.
        """
        from pgqueuer.models import Job

        @self._qm.entrypoint(ENTRYPOINT)
        async def _run(pg_job: Job) -> None:
            await handler(Batch([IngestionJob.model_validate_json(pg_job.payload)]))

    async def run(self) -> None:
        await self._qm.run()
