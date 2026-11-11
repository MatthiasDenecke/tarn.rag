"""IngestionEngine — the high-level facade for the ingestion pipeline.

The public surface is **document-centric**: callers submit documents and poll by
``document_id`` (== ``source_id``). Jobs are an internal detail and never leak here — the
only window is the debug-gated :meth:`document_jobs` (raises unless ``APP__DEBUG``).

Use :meth:`create` for ready-to-use wiring from ``Settings``:

* ``MODE='embedded'`` (default) runs the whole pipeline **in-process** (InMemory queue, no
  Postgres/pgQueuer) — each ingest call processes to completion before returning.
* ``MODE='distributed'`` enqueues to pgQueuer; separate worker processes (``run_worker``)
  consume the queue.

The bare constructor is the low-level seam (inject your own pipeline/orchestrator — used by
tests and advanced wiring), mirroring ``RetrievalEngine.open``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from tarnrag.contracts import DocumentFactsSource, IndexMeta, PipelineItem
from tarnrag.core.engine.config import (
    INGESTION_PIPELINE,
    IdPolicy,
    Settings,
    get_settings,
)
from tarnrag.core.engine.engine import Engine
from tarnrag.core.engine.observability import Observability
from tarnrag.core.exceptions import IngestionError
from tarnrag.core.hashing import sha256_file, sha256_hex
from tarnrag.core.resources.embedder import Embedder
from tarnrag.ingestion.components.extraction.load_parse import infer_source_kind
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import (
    InMemoryJobQueue,
    JobConsumer,
    JobEnqueuer,
    PgQueuerJobQueue,
)
from tarnrag.ingestion.engine.result_sink import create_sink_registry
from tarnrag.ingestion.engine.types import DocumentStatus, DocumentSummary
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.pipeline import Pipeline
from tarnrag.storage.repository import DocumentRepository
from tarnrag.storage.status import DocumentStatusReader

logger = logging.getLogger(__name__)


class IngestionEngine(Engine):
    """
    High-level facade for the ingestion pipeline; document-centric public surface.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        orchestrator: PipelineOrchestrator,
        repository: DocumentRepository,
        observability: Any = None,  # Phase 5 Observability plugs in here
        staging_dir: str | None = None,  # where streamed bytes are staged for the worker
        facts_source: DocumentFactsSource | None = None,  # data store for status counts
        *,
        queue: JobEnqueuer | None = None,  # held for embedded draining / the worker loop
        auto_drain: bool = False,  # embedded: process the whole DAG inline on each ingest
        debug: bool = False,  # gates the debug-only methods (set from APP__DEBUG)
        id_policy: IdPolicy = "uuid",  # how document ids are assigned (set from ID_POLICY)
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + job_status projection
        self.obs = observability
        self.staging_dir = staging_dir
        # The document data store — the repository, which now also holds the §8 retrieval index
        # (one store): it supplies status facts AND the content_hash dedup lookup. The seam stays
        # (a narrow DocumentFactsSource port) but defaults to the repository.
        self._facts_source = facts_source or repository
        self._status = DocumentStatusReader(repository, self._facts_source)
        self._queue = queue
        self._auto_drain = auto_drain
        self._debug = debug
        self._id_policy = id_policy
        # Serializes embedded submissions (enqueue + drain, see ``_submit``): two concurrent ingest() calls
        # would otherwise interleave two run() loops over the one queue, or mix two runs into one wave.
        self._drain_lock = asyncio.Lock()

    # ----- construction -----

    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        repository: DocumentRepository | None = None,
        embedder: Embedder | None = None,
    ) -> IngestionEngine:
        """Build a ready-to-use engine from ``Settings`` (mode-driven). The easy entry point; the
        constructor is the lower-level seam for injecting your own wiring. The pipeline composition
        comes from ``Settings.components`` (see :meth:`build_pipeline`).

        A composition root (e.g. ``TarnRag``) may inject a pre-built ``repository`` + ``embedder`` so
        one store + embedder is shared across the engines instead of each building its own — pass both
        or neither. The ``embedder`` is used only to stamp the index identity (``write_index_meta``);
        the Embed stage builds its own from ``Settings``."""
        settings = settings or get_settings()
        obs = Observability.create(settings.observability)
        if repository is None and embedder is None:
            repository = await DocumentRepository.create(settings.database, settings.EMBEDDING_DIMENSION)
            embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
        elif repository is None or embedder is None:
            raise ValueError("inject both repository and embedder, or neither")
        # The engine is the index producer: establish/confirm the §8 build/identity record (the sinks
        # then persist document/chunk/embedding data into the same repository; job_status lives there too).
        await cls.ensure_index_meta(repository, embedder)
        pipeline = cls.build_pipeline(settings)

        if settings.MODE == "distributed":
            queue: JobEnqueuer = await PgQueuerJobQueue.connect(settings.database.queue_url)
            auto_drain = False
        else:  # embedded — run the whole pipeline in this process
            queue = InMemoryJobQueue()
            auto_drain = True

        orchestrator = PipelineOrchestrator(
            PipelineDAG(pipeline.stages), queue, repository, create_sink_registry(),
            observability=obs,  # chunk_store defaults to the repository
        )
        engine = cls(
            pipeline, orchestrator, repository, observability=obs,
            staging_dir=settings.UPLOAD_DIR,  # facts_source defaults to the repository
            queue=queue, auto_drain=auto_drain, debug=settings.app.debug,
            id_policy=settings.ID_POLICY,
        )
        if auto_drain:  # register the in-process worker that ingest() will drive
            _attach_worker(queue, orchestrator, obs)
        return engine

    @staticmethod
    async def ensure_index_meta(repository: DocumentRepository, embedder: Embedder) -> None:
        """ESTABLISH the §8 build/identity record on a fresh index, or CONFIRM it on an existing one —
        never silently re-stamp (that would write on a read-only open and mask an embedder mismatch).
        Cheap (repository-only), so a composition root can call it without building the full engine;
        ``RetrievalEngine.open`` validates the same record."""
        existing = await repository.index_meta()
        if not existing.get("schema_version"):
            await repository.write_index_meta(IndexMeta.build(embedder))
        elif (conflict := IndexMeta.conflict(existing, embedder)) is not None:
            raise IngestionError(conflict)

    @staticmethod
    def build_pipeline(settings: Settings) -> Pipeline:
        """Build the ingestion pipeline from ``Settings`` — the spec at
        ``settings.components[INGESTION_PIPELINE]`` (``Settings`` guarantees it is present, default or
        user-supplied). The Embed stage's embedding identity is always taken from ``Settings`` (not
        the spec), so a pipeline spec can't drift from the retrieval-side fingerprint."""
        spec = IngestionEngine._with_embedding_from_settings(
            settings.components[INGESTION_PIPELINE], settings
        )
        return Pipeline.from_spec(spec)

    @staticmethod
    def _with_embedding_from_settings(spec: dict[str, Any], settings: Settings) -> dict[str, Any]:
        """Return a copy of ``spec`` with every Embed stage's ``embedding`` / ``embedding_dimension``
        set from ``settings`` — the embedding identity is cross-cutting (shared with retrieval)."""
        stages = []
        for stage_spec in spec["stages"]:
            stage_spec = dict(stage_spec)
            if stage_spec.get("class_name") == "Embed":
                stage_spec["embedding"] = settings.embedding.model_dump()
                stage_spec["embedding_dimension"] = settings.EMBEDDING_DIMENSION
            stages.append(stage_spec)
        return {**spec, "stages": stages}

    # ----- ingest -----

    async def ingest_paths(
        self, paths: list[str], *, source_ids: list[str] | None = None, extractor: str | None = None
    ) -> list[str]:
        """Ingest documents loaded from file paths (LoadAndParse reads them). ``extractor`` overrides
        the configured extractor for these documents (e.g. ``'docling'``; None → the LoadAndParse
        stage's configured route for the detected source kind).

        ``source_ids`` (parallel to ``paths``) sets each document's ``source_id`` (==
        ``document_id``), which is **stable**: re-ingesting the same id upserts in place. Under
        ``ID_POLICY='caller'`` it is required; under ``'uuid'`` it must be omitted (the engine
        assigns ids). Content dedup is separate — every document stores a ``content_hash``
        (query :meth:`find_by_content_hash`). Returns the document IDs (poll :meth:`status`)."""
        ids = self._resolve_document_ids(source_ids, len(paths))
        # Hash off the event loop — file hashing is blocking I/O (large files would stall the loop).
        hashes = await asyncio.gather(*(asyncio.to_thread(sha256_file, path) for path in paths))
        items = [
            self._item(
                source_id=ids[i],
                content="",  # loaded by LoadAndParseStage
                extra={
                    "source_path": path,
                    "source_type": infer_source_kind(path),
                    "content_hash": hashes[i],  # dedup key (sha256 of the file bytes)
                    **({"extractor": extractor} if extractor else {}),
                },
            )
            for i, path in enumerate(paths)
        ]
        return await self._submit(items)

    async def ingest_content(
        self, documents: list[dict[str, str]], *, extractor: str | None = None
    ) -> list[str]:
        """Ingest pre-loaded content. Each doc's ``source_id`` is governed by ``ID_POLICY``
        (required under ``'caller'``, forbidden under ``'uuid'``). The document's
        ``content_hash`` is the sha256 of its ``content``. Returns the document IDs."""
        docs = [dict(d) for d in documents]  # don't mutate the caller's dicts
        contents = [d.pop("content") for d in docs]
        ids = self._resolve_document_ids([d.pop("source_id", None) for d in docs], len(docs))
        items = []
        for i, doc in enumerate(docs):
            if extractor:
                doc.setdefault("extractor", extractor)
            doc["content_hash"] = sha256_hex(contents[i])
            items.append(self._item(ids[i], content=contents[i], extra=doc))
        return await self._submit(items)

    async def ingest_streams(
        self, streams: list[tuple[str, BinaryIO]], *, source_ids: list[str] | None = None,
        extractor: str | None = None,
    ) -> list[str]:
        """Ingest binary streams (e.g. HTTP uploads): **stream** each ``(filename, source)``
        to the staging dir, then ingest by path (parsing happens in the worker). ``source`` is
        any binary file-like; it is copied in chunks (never fully held in memory). Requires
        ``staging_dir``; in ``distributed`` mode that must be a volume the worker can read.

        ``source_ids`` (parallel to ``streams``) follows ``ID_POLICY`` exactly as in
        :meth:`ingest_paths`; the ``content_hash`` is the sha256 of the streamed bytes.
        Returns the document IDs."""
        if not self.staging_dir:
            raise RuntimeError("streams are not configured (no staging_dir / UPLOAD_DIR)")
        # Validate the id supply against the policy BEFORE staging, so a bad call leaves no
        # orphaned files; ingest_paths then resolves ids + hashes from the staged files.
        self._check_supply(self._align_source_ids(source_ids, len(streams)))
        # Offload the blocking disk copy so the event loop isn't stalled on large files.
        paths = [
            await asyncio.to_thread(self._stage_stream, filename, source)
            for filename, source in streams
        ]
        return await self.ingest_paths(paths, source_ids=source_ids, extractor=extractor)

    async def status(self, document_id: str) -> DocumentStatus | None:
        """Document-level status derived from persisted data (pending | in_progress |
        complete | failed). ``None`` if the document is unknown."""
        raw = await self._status.document_status(document_id)
        if raw is None:
            return None
        return DocumentStatus(
            document_id=raw["document_id"],
            status=raw["status"],
            chunk_count=raw["chunk_count"],
            embedding_count=raw["embedding_count"],
        )

    async def find_by_content_hash(self, content_hash: str) -> list[str]:
        """Document IDs whose stored content matches ``content_hash`` — the content-dedup query.
        Empty if that content was never ingested; possibly several (same content under different
        ids). Build the key with :meth:`content_hash` / :meth:`content_hash_of_file`. Identity is
        independent: a document keeps its id when its content (and this hash) changes."""
        return await self._facts_source.documents_by_content_hash(content_hash)

    @staticmethod
    def content_hash(data: bytes) -> str:
        """The sha256 hex the engine stores for submitted content — use it to build a dedup key
        (e.g. ``await engine.find_by_content_hash(engine.content_hash(data))``)."""
        return sha256_hex(data)

    @staticmethod
    def content_hash_of_file(path: str) -> str:
        """:meth:`content_hash` for a file, read in chunks (constant memory)."""
        return sha256_file(path)

    async def list_documents(self) -> list[DocumentSummary]:
        """Inventory of every ingested document — id, ``content_hash``, and chunk/embedding
        counts. Order is unspecified; pair with :meth:`status` for per-document state."""
        return [
            DocumentSummary(
                document_id=r["document_id"],
                content_hash=r["content_hash"],
                chunk_count=r["chunk_count"],
                embedding_count=r["embedding_count"],
            )
            for r in await self._facts_source.list_documents()
        ]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and everything derived from it — chunks, embeddings, retrieval-index
        rows, and its job_status records — in a **single atomic transaction**. Returns True if the
        document was known (had data or in-flight jobs), False if there was nothing to delete.

        The data and the job_status projection live in the one repository, so this is all-or-
        nothing: a failure rolls everything back — never a half-deleted document or a 'ghost'
        status pointing at deleted data. Idempotent: deleting an unknown document is a no-op."""
        return await self.repository.delete_document_and_jobs(document_id)

    async def aclose(self) -> None:
        """Release what this engine built: the queue (the pgQueuer asyncpg pool in distributed mode —
        the InMemory queue holds nothing) and then the repository (the base ``Engine`` lifecycle)."""
        closer = getattr(self._queue, "aclose", None)
        if closer is not None:
            await closer()
        await super().aclose()

    # ----- debug (gated on APP__DEBUG) -----

    async def document_jobs(self, document_id: str) -> list[dict]:
        """Per-job breakdown for a document — the one window into the otherwise-internal job
        state. **Debug only:** raises ``RuntimeError`` unless ``APP__DEBUG`` is set."""
        self._require_debug()
        return await self.repository.document_jobs(document_id)

    # ----- internals -----

    def _require_debug(self) -> None:
        if not self._debug:
            raise RuntimeError("debug-only operation; set APP__DEBUG=true to enable")

    def _resolve_document_ids(
        self, source_ids: list[str | None] | None, count: int
    ) -> list[str]:
        """Document ids per ``ID_POLICY``: 'caller' uses the supplied ids (every one required);
        'uuid' assigns fresh uuid4s (and forbids supplied ids). Raises on a policy mismatch."""
        supplied = self._align_source_ids(source_ids, count)
        self._check_supply(supplied)
        if self._id_policy == "caller":
            return [s for s in supplied]  # all present (validated by _check_supply)
        return [str(uuid.uuid4()) for _ in range(count)]

    def _align_source_ids(
        self, source_ids: list[str | None] | None, count: int
    ) -> list[str | None]:
        """Normalize caller ids to one slot per document (None where unset); validate length."""
        if source_ids is None:
            return [None] * count
        if len(source_ids) != count:
            raise ValueError(
                f"source_ids has {len(source_ids)} entries but {count} documents were given"
            )
        return list(source_ids)

    def _check_supply(self, supplied: list[str | None]) -> None:
        """Validate the caller's id supply against ``ID_POLICY`` (presence/absence)."""
        count = len(supplied)
        given = sum(1 for s in supplied if s)
        if self._id_policy == "caller" and given != count:
            raise ValueError(
                f"ID_POLICY='caller' requires a source_id for every document "
                f"({count - given} of {count} missing)"
            )
        if self._id_policy == "uuid" and given:
            raise ValueError(
                f"ID_POLICY='uuid' assigns document ids automatically; remove the "
                f"{given} caller-supplied source_id(s) (or set ID_POLICY='caller')"
            )

    async def _submit(self, items: list[PipelineItem]) -> list[str]:
        if not (self._auto_drain and isinstance(self._queue, JobConsumer)):
            return await self._enqueue_documents(items)  # distributed: the workers process them
        # Embedded: process the whole DAG in-process, one submission at a time. The lock covers the
        # enqueue as well as the drain, so a concurrent ingest waits for this one to finish instead of
        # adding its jobs to the running waves. Otherwise two runs of one document could share a batch
        # (two chunk sets in one store_chunks call), and its job-history clear would wipe a run in flight.
        async with self._drain_lock:
            document_ids = await self._enqueue_documents(items)
            await self._queue.run()
        return document_ids

    async def _enqueue_documents(self, items: list[PipelineItem]) -> list[str]:
        document_ids = await self.orchestrator.ingest_documents(items)
        if self.obs:
            self.obs.counter("ingestion.documents_queued", len(items))
            await self.obs.log("info", "documents queued", count=len(document_ids))
        return document_ids

    def _item(self, source_id: str, content: str, extra: dict[str, Any]) -> PipelineItem:
        return PipelineItem(
            id=source_id,
            content=content,
            metadata={
                "source_id": source_id,  # public handle == document_id
                "created_at": datetime.now(UTC).isoformat(),
                **extra,
            },
        )

    def _stage_stream(self, filename: str, source: BinaryIO) -> str:
        """Stream a binary file-like under a unique name (extension preserved so the loader
        dispatches correctly) and return the path. ``copyfileobj`` copies in chunks — no
        full-memory load. An object store (S3/blob) would plug in behind this method."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else "bin"
        dest = Path(self.staging_dir) / f"{uuid.uuid4()}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            shutil.copyfileobj(source, out)
        return str(dest)


def _attach_worker(queue: JobConsumer, orchestrator: PipelineOrchestrator, obs: Any) -> IngestionWorker:
    """Build the in-process worker and register it as the queue's handler; return it. Shared by the
    embedded ``create`` (auto-drain) and the distributed ``run_worker`` entry point."""
    worker = IngestionWorker(orchestrator, observability=obs)
    queue.set_handler(worker.handle_batch)
    return worker


async def run_worker(settings: Settings | None = None) -> None:
    """Consumer entry point for ``MODE='distributed'`` — build the wiring via
    :meth:`IngestionEngine.create` and run the consume loop until the queue stops, then release
    resources. (In ``embedded`` mode ingestion runs in-process, so there is no separate worker.)
    A distributed worker must use the SAME ``Settings`` — incl. ``components`` — as the producer."""
    engine = await IngestionEngine.create(settings)
    if not isinstance(engine._queue, JobConsumer):
        raise TypeError("this engine has no consumable queue (a JobConsumer) to run a worker on")
    worker = _attach_worker(engine._queue, engine.orchestrator, engine.obs)
    logger.info("Ingestion worker %s started", worker.worker_id)
    try:
        await engine._queue.run()
    finally:
        await engine.aclose()
