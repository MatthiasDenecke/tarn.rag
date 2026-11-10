"""Ingestion job + batch models — internal units of work, never exposed to API clients."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from tarnrag.contracts import PipelineItem


class IngestionJob(BaseModel):
    """
    Internal unit of work enqueued in the job queue (pgQueuer in prod).

    Runtime state (queued/processing/.../retries) is owned by the queue; for the
    status API a document-keyed ``job_status`` projection is kept in the repository.
    The stage to run is identified by ``stage_name``; the worker gets the configured
    stage instance from the coordinator's DAG (no per-job reconstruction).
    """

    job_id: str  # logical id (used by the status projection)
    document_id: str  # public document handle (== source_id)
    item: PipelineItem  # in-flight item, INLINE in the payload (D1)
    stage_name: str  # stage to execute
    created_at: datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Batch:
    """
    A homogeneous dispatch unit: a list of jobs that all target the SAME stage.

    This is what the queue (``JobConsumer``) hands the worker, and the compute batch a
    stage runs in one shot. Same-stage homogeneity is the invariant that makes a batch
    *meaningful* — a single stage runs the whole batch, and two-tier batching (D2) only
    batches compute *within* a stage. A mixed-stage batch is therefore not a thing.

    The invariant is enforced HERE, at construction, so callers (the worker, the
    coordinator) can rely on a single ``stage_name`` without re-checking. ``stage_name``
    stays on each ``IngestionJob`` (the queue routes by it); ``Batch`` derives the batch's
    stage from its jobs and refuses a mismatch. Forming Batches is the queue's job — when
    batch dispatch claims several jobs, it must group them so each Batch is homogeneous.

    ``final_attempt`` is the queue's word on retries: ``False`` when a failure of this batch will be
    retried (the queue still has attempts left for it), ``True`` when this is the last try — so the
    status projection can tell a transient failure (the job is queued again) from a terminal one.
    """

    __slots__ = ("final_attempt", "jobs", "stage_name")

    def __init__(self, jobs: list[IngestionJob], *, final_attempt: bool = True):
        """
        Build a batch from same-stage jobs, enforcing the homogeneity invariant (raises on an
        empty list or a mix of ``stage_name``s).
        """
        if not jobs:
            raise ValueError("Batch must contain at least one job")
        stage_names = {job.stage_name for job in jobs}
        if len(stage_names) > 1:
            raise ValueError(
                f"Batch jobs must all share one stage_name; got {sorted(stage_names)}"
            )
        self.jobs = jobs
        self.stage_name = jobs[0].stage_name
        self.final_attempt = final_attempt
