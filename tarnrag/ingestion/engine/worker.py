"""Ingestion worker — the job HANDLER. COMPUTE ONLY (D2/D3)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import nullcontext
from typing import Any

from tarnrag.ingestion.engine.batch import BatchCoordinator
from tarnrag.ingestion.engine.jobs import Batch

logger = logging.getLogger(__name__)


class IngestionWorker:
    """
    The job handler — COMPUTE ONLY (D2/D3). A pure handler: it holds only the
    coordinator and exposes ``handle_batch``. It does not touch the queue — the
    composition root registers ``worker.handle_batch`` with a ``JobConsumer`` and runs
    the consumer's loop.

    For each dispatched batch it gets the stage from the coordinator, runs it, reports
    produced results to the ``BatchContext``, and then ``complete``s it (or ``fail``s on
    compute error and re-raises). It never touches the repo, queue, registry, or
    ResultSink. A raised exception propagates to the consumer, which requeues (D5).

    Stages are synchronous and do real work — file reads, PDF layout parsing, ONNX inference, blocking
    HTTP to an embedding API — so the compute runs in a worker thread (``asyncio.to_thread``): the event
    loop keeps serving the queue's heartbeats / NOTIFY listener and any retrieval sharing the process
    while a batch computes.
    """

    def __init__(
        self,
        coordinator: BatchCoordinator,
        observability: Any = None,  # Phase 5 Observability plugs in here
        worker_id: str | None = None,
    ):
        self.coordinator = coordinator
        self.obs = observability
        self.worker_id = worker_id or str(uuid.uuid4())[:8]

    async def handle_batch(self, batch: Batch) -> None:
        """Process one dispatched ``Batch`` (homogeneous: one stage). Raising → requeue
        (recovery); returning normally → ack."""
        stage_name  = batch.stage_name
        tag         = stage_name  # type tag for metrics; resolved from the stage instance below
        ctx         = await self.coordinator.begin_batch(batch)
        produced    = 0
        try:
            stage   = self.coordinator.get_stage(stage_name)  # long-lived DAG instance
            tag     = stage.tag
            items   = [job.item for job in batch.jobs]  # inline items (D1)
            # Time the pure compute (stages stay obs-free; the worker observes them).
            timer   = self.obs.timer(f"stage.{tag}.process") if self.obs else nullcontext()
            with timer:  # off the event loop: the stage generator is drained in a worker thread
                results = await asyncio.to_thread(lambda: list(stage.process_batch(items)))
            for result in results:
                # Re-id items that FLOW onward (a PipelineItem with its id reset). A result
                # without an ``id`` field at all — the terminal ``Embedding``, keyed by
                # chunk_id — has no identity to assign (getattr default: absent ≠ unset).
                if getattr(result, "id", "") is None:
                    result.id = str(uuid.uuid4())
                ctx.submit([result])
                produced += 1
            if self.obs:
                self.obs.counter(f"stage.{tag}.batches")
                self.obs.counter(f"stage.{tag}.items", produced)
        except Exception as e:
            logger.exception("Worker %s compute failed on %s", self.worker_id, stage_name)
            if self.obs:
                self.obs.counter(f"stage.{tag}.errors")
                await self.obs.log(
                    "error", "stage compute failed",
                    stage=tag, worker=self.worker_id, error=str(e),
                )
            await ctx.fail(e)
            raise
        # Finalize + DAG advance belong to the context (D5); raises on persist failure.
        await ctx.complete()
