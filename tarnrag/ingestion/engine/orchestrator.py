"""Pipeline DAG + orchestrator — walks the DAG and owns the job lifecycle (D3/D5).

The orchestrator is the ``BatchCoordinator``: it enqueues root jobs, and for each
dispatched batch it hands the worker a ``BatchContext`` that owns persistence + status
+ DAG advance. The worker reports to that context and never holds the orchestrator.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from tarnrag.core.exceptions import IngestionError
from tarnrag.contracts import ChunkStore, PipelineItem
from tarnrag.storage.repository import DocumentRepository
from tarnrag.ingestion.engine.batch import BatchContext, BatchCoordinator
from tarnrag.ingestion.engine.jobs import Batch, IngestionJob
from tarnrag.ingestion.engine.queue import JobEnqueuer
from tarnrag.ingestion.engine.result_sink import ResultSink

logger = logging.getLogger(__name__)


class PipelineDAG:
    """
    The pipeline as a directed acyclic graph. Sequential edges for now:
    ``stage[i] -> stage[i+1]``.

    Also the **naming authority**: node names are the routing identity — every queued job carries a
    ``stage_name`` that the *consumer's* DAG resolves (in distributed mode, a different process) — so
    they must be derivable from the pipeline definition alone. An unnamed stage is assigned
    ``{tag}-{position}`` here (same ``Settings`` ⇒ same names in every process, every restart); an
    explicit ``config.name`` wins. Duplicate names are refused up front — ``get_stage`` would silently
    route every job to the first match. A stage instance belongs to one pipeline: re-using it in a
    second DAG at a different position would rename it out from under the first DAG's edges.
    """

    def __init__(self, stages: list):
        for i, stage in enumerate(stages):
            if getattr(stage.config, "name", None) is None:
                stage._name = f"{stage.tag}-{i}"  # positional — no process-local state in the identity
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate stage name(s) {dupes} — every DAG node needs a unique name")
        self.stages = stages
        self.edges = [(stages[i].name, stages[i + 1].name) for i in range(len(stages) - 1)]

    def get_downstream_stages(self, stage_name: str) -> list[str]:
        """Stage names that depend on this stage (empty for the terminal stage)."""
        return [target for source, target in self.edges if source == stage_name]

    def get_stage(self, stage_name: str):
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        raise ValueError(f"Stage {stage_name} not found")


class PipelineOrchestrator(BatchCoordinator):
    """
    Walks the DAG and owns the JOB LIFECYCLE (D3/D5).

    The queue owns claiming/retries/acking; the orchestrator enqueues jobs through
    it, and per dispatched batch it issues a ``BatchContext`` (via ``begin_batch``) that
    finalizes the sink, records status, and enqueues downstream jobs — or raises so the
    queue requeues (recovery). Also keeps the document-keyed ``job_status`` projection.
    """

    def __init__(
        self,
        dag: PipelineDAG,
        enqueuer: JobEnqueuer,
        repository: DocumentRepository,
        sink_registry: dict[str, type[ResultSink]],
        observability: Any = None,  # lifecycle metrics (enqueue/status/persistence)
        chunk_store: ChunkStore | None = None,  # sink target; defaults to the repository
    ):
        self.dag = dag
        self.enqueuer = enqueuer
        self.repository = repository  # job_status projection (record_job)
        self.obs = observability
        self.sink_registry = sink_registry
        # Where sinks persist document/chunk/embedding data — the repository, which holds the §8
        # retrieval index (sqlite-vec/FTS5 or pgvector). The param lets a test inject another store.
        self.chunk_store = chunk_store or repository

    async def ingest_documents(self, items: list[PipelineItem]) -> list[str]:
        """Enqueue one root job per document for the first stage. Returns the public
        document_ids (== source_ids); job ids never leave the system.

        A (re-)submission starts the document's job history afresh: the job_status rows of any earlier
        ingest of the same id are cleared first, so a stale ``failed`` row from a run that has since been
        superseded can't pin the document's status to ``failed`` forever."""
        first_stage = self.dag.stages[0].name
        document_ids = []
        for item in items:
            job = self._make_job(item, first_stage)
            await self.repository.delete_document_jobs(job.document_id)
            await self._enqueue(job)
            document_ids.append(job.document_id)
        if self.obs:
            self.obs.counter("ingest.documents", len(items))
        return document_ids

    def get_stage(self, stage_name: str):
        """The long-lived, configured stage instance from the DAG (reused across jobs)."""
        return self.dag.get_stage(stage_name)

    async def begin_batch(self, batch: Batch) -> BatchContext:
        """Start a unit of work for a dispatched batch: record 'processing', pick the
        per-stage sink, and return the context the worker reports to."""
        await self._record(batch.jobs, "processing")
        tag = self.dag.get_stage(batch.stage_name).tag  # sinks + metrics dispatch on the type tag
        sink = self.sink_registry[tag](self.chunk_store)
        return _OrchestratorBatchContext(self, batch, sink, tag)

    # ----- internals shared by ingest_documents and the BatchContext -----

    async def _record(
        self, jobs: list[IngestionJob], status: str, error: str | None = None
    ) -> None:
        await self.repository.record_jobs(  # one transaction for the whole batch's status rows
            [(job.document_id, job.job_id, job.stage_name) for job in jobs], status, error
        )

    def _make_job(self, item: PipelineItem, stage_name: str) -> IngestionJob:
        return IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=item.metadata["source_id"],  # public handle; threads via metadata
            item=item,
            stage_name=stage_name,
            created_at=datetime.now(UTC),
        )

    async def _enqueue(self, job: IngestionJob) -> None:
        await self.repository.record_job(
            job.document_id, job.job_id, job.stage_name, "queued"
        )
        await self.enqueuer.enqueue(job)
        if self.obs:
            self.obs.counter("ingest.jobs_enqueued")


class _OrchestratorBatchContext(BatchContext):
    """
    Per-batch unit of work backed by the orchestrator's repo / DAG / queue. It
    composes the per-stage ResultSink (kept pure) and adds status + DAG advance.
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        batch: Batch,
        sink: ResultSink,
        tag: str,
    ):
        self._orch = orchestrator
        self._batch = batch
        self._sink = sink
        self._tag = tag  # the stage's type tag (the obs-metric key)
        self._produced: list[Any] = []

    def submit(self, results: list[Any]) -> None:
        self._produced.extend(results)
        self._sink.submit(results)

    async def fail(self, error: BaseException) -> None:
        await self._orch._record(self._batch.jobs, "failed", str(error))
        if self._orch.obs:
            self._orch.obs.counter(f"stage.{self._tag}.failed")

    async def complete(self) -> None:
        obs = self._orch.obs
        self._sink.close()
        outcome = await self._sink.finalize()
        if not outcome.persisted:
            await self._orch._record(self._batch.jobs, "failed", f"persistence: {outcome.detail}")
            if obs:
                obs.counter(f"stage.{self._tag}.persist_failed")
                await obs.log(
                    "error", "persistence failed",
                    stage=self._tag, detail=outcome.detail,
                )
            raise IngestionError(
                f"persistence failed for {self._tag}: {outcome.detail}"
            )
        await self._orch._record(self._batch.jobs, "completed")
        if obs:
            obs.counter(f"stage.{self._tag}.completed")
        # Fan-out (D1): one downstream job per produced item, per next stage.
        for next_stage in self._orch.dag.get_downstream_stages(self._batch.stage_name):
            for item in self._produced:
                await self._orch._enqueue(self._orch._make_job(item, next_stage))
