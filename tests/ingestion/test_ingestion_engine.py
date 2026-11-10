"""IngestionEngine facade on real InMemoryJobQueue + SQLite wiring (no API, no Postgres).

Uses a fake 3-d encoder so no sentence-transformers is needed (matches the e2e test). The
bare constructor is used here (the low-level seam); ``create()`` is exercised only when a real
model + index are available.
"""

import io
from pathlib import Path

import pytest

from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue, JobEnqueuer
from tarnrag.ingestion.engine.result_sink import create_sink_registry
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.embed import EmbedStage
from tarnrag.ingestion.pipeline.pipeline import Pipeline


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


def _wire(repo, *, auto_drain=False, debug=False, policy="caller"):
    """Build producer+consumer wiring around one repo; return (engine, queue)."""
    queue = InMemoryJobQueue()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)
    engine = IngestionEngine(
        Pipeline.from_stages(stages), orch, repo, queue=queue, auto_drain=auto_drain, debug=debug,
        id_policy=policy,
    )
    return engine, queue


async def test_ingest_content_queues_then_completes(repo):
    engine, queue = _wire(repo)

    result = await engine.ingest_content([{"content": "Hello world. " * 10, "source_id": "s1"}])
    assert result == ["s1"]  # ingest returns the document IDs
    # Queued but not yet processed -> pending (no persisted data).
    assert (await engine.status("s1")).status == "pending"

    await queue.run()  # drain the whole DAG

    status = await engine.status("s1")
    assert status.status == "complete"
    assert status.chunk_count >= 1
    assert status.embedding_count == status.chunk_count


async def test_embedded_auto_drain_processes_inline(repo):
    """With auto_drain (embedded mode) an ingest call runs the whole pipeline before returning."""
    engine, _ = _wire(repo, auto_drain=True)
    await engine.ingest_content([{"content": "Hello world. " * 10, "source_id": "s1"}])
    # No manual queue.run() — auto_drain processed it inline.
    status = await engine.status("s1")
    assert status.status == "complete"
    assert status.embedding_count == status.chunk_count >= 1


async def test_concurrent_ingests_single_flight_the_drain(repo):
    """Two+ concurrent embedded ingests must not interleave two ``run()`` loops over the one queue (waves
    and batches would be split arbitrarily between them): the drain lock serializes them, and every
    document still completes."""
    import asyncio

    engine, queue = _wire(repo, auto_drain=True)
    active = {"now": 0, "max": 0}
    original_run = queue.run

    async def tracked_run():
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        try:
            await original_run()
        finally:
            active["now"] -= 1

    queue.run = tracked_run
    batches = [[{"content": f"Document {i} text. " * 10, "source_id": f"d{i}"}] for i in range(4)]
    results = await asyncio.gather(*(engine.ingest_content(b) for b in batches))

    assert active["max"] == 1  # never two drain loops at once
    for [doc_id] in results:  # a drain may have processed a peer's jobs — every doc still completes
        assert (await engine.status(doc_id)).status == "complete"


async def test_aclose_closes_a_closable_queue(repo):
    """``aclose`` releases the queue the engine was built with (the pgQueuer pool in distributed mode)
    before the repository — through the same duck-typed ``aclose`` seam the facade uses."""
    engine, queue = _wire(repo)
    closed = []

    async def record():
        closed.append("queue")

    queue.aclose = record  # the InMemory queue has none; give this instance a recording one
    await engine.aclose()
    assert closed == ["queue"]


async def test_aclose_without_a_closable_queue_still_releases_the_store(repo):
    """The InMemory queue holds nothing to close — ``aclose`` skips it and still releases the store."""
    engine, _ = _wire(repo)
    await engine.aclose()  # no error; repository disconnected (idempotent under the fixture's teardown)


async def test_unknown_document_status_is_none(repo):
    engine, _ = _wire(repo)
    assert await engine.status("does-not-exist") is None


async def test_document_jobs_is_debug_gated(repo):
    engine, queue = _wire(repo)  # debug off (default)
    await engine.ingest_content([{"content": "alpha beta gamma. " * 8, "source_id": "s1"}])
    await queue.run()

    # debug off -> the per-job breakdown is refused
    with pytest.raises(RuntimeError, match="APP__DEBUG"):
        await engine.document_jobs("s1")

    # debug on (same repo) -> returns the breakdown
    engine_dbg, _ = _wire(repo, debug=True)
    jobs = await engine_dbg.document_jobs("s1")
    assert isinstance(jobs, list) and jobs
    assert all(j["status"] == "completed" for j in jobs)


async def test_uuid_policy_assigns_ids(repo):
    engine, _ = _wire(repo, policy="uuid")
    result = await engine.ingest_content([{"content": "x"}])
    assert len(result) == 1
    assert result[0]  # a uuid was assigned


class _RecordingEnqueuer(JobEnqueuer):
    def __init__(self):
        self.jobs: list = []

    async def enqueue(self, job) -> None:
        self.jobs.append(job)


async def test_extractor_override_rides_in_item_metadata(repo):
    """The per-request extractor override flows inline on the root job's item (not stage config)."""
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(Pipeline.from_stages(stages), orch, repo, id_policy="caller")

    await engine.ingest_content([{"content": "x", "source_id": "s1"}], extractor="docling")
    assert enq.jobs[0].item.metadata["extractor"] == "docling"

    # Omitted -> no extractor key (the stage uses its configured route).
    enq.jobs.clear()
    await engine.ingest_content([{"content": "x", "source_id": "s2"}])
    assert "extractor" not in enq.jobs[0].item.metadata


async def test_ingest_streams_stages_bytes_and_queues_by_path(repo, tmp_path):
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(Pipeline.from_stages(stages), orch, repo, staging_dir=str(tmp_path / "uploads"))

    result = await engine.ingest_streams(
        [("report.pdf", io.BytesIO(b"%PDF-fake-bytes"))], extractor="docling"
    )
    assert len(result) == 1

    meta = enq.jobs[0].item.metadata
    staged = meta["source_path"]
    assert staged.endswith(".pdf")  # extension preserved so the loader dispatches
    assert Path(staged).read_bytes() == b"%PDF-fake-bytes"  # streamed to disk
    assert meta["source_type"] == "pdf"
    assert meta["extractor"] == "docling"


async def test_ingest_streams_requires_staging_dir(repo):
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), InMemoryJobQueue(), repo, create_sink_registry())
    engine = IngestionEngine(Pipeline.from_stages(stages), orch, repo)  # no staging_dir
    with pytest.raises(RuntimeError, match="not configured"):
        await engine.ingest_streams([("x.txt", io.BytesIO(b"hi"))])


async def test_ingest_paths_uses_supplied_source_ids(repo, tmp_path):
    """source_ids set the document ids (== source_id) and align one-to-one with paths."""
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(Pipeline.from_stages(stages), orch, repo, id_policy="caller")

    a = tmp_path / "a.txt"
    a.write_text("aaa", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("bbb", encoding="utf-8")
    result = await engine.ingest_paths([str(a), str(b)], source_ids=["id-a", "id-b"])
    assert result == ["id-a", "id-b"]  # returned document ids are the supplied ones
    # Each supplied id rides on its matching path's root job.
    by_id = {j.item.metadata["source_id"]: j.item.metadata["source_path"] for j in enq.jobs}
    assert by_id == {"id-a": str(a), "id-b": str(b)}


async def test_ingest_paths_persists_the_source_kind(repo, tmp_path):
    """The extractor-confirmed kind lands in the document row. It used to fall back to ``'document'``
    for every ingest — the engine stamped ``source_type`` while the repository read ``source_kind``."""
    f = tmp_path / "spec.txt"
    f.write_text("Hello world. " * 10, encoding="utf-8")
    engine, queue = _wire(repo)
    await engine.ingest_paths([str(f)], source_ids=["spec"])
    await queue.run()
    doc = await repo.get_document("spec")
    assert doc.metadata["source_kind"] == "txt"  # the real kind, not the 'document' default


async def test_ingest_streams_uses_supplied_source_ids(repo, tmp_path):
    enq = _RecordingEnqueuer()
    stages = _stages()
    orch = PipelineOrchestrator(PipelineDAG(stages), enq, repo, create_sink_registry())
    engine = IngestionEngine(
        Pipeline.from_stages(stages), orch, repo, staging_dir=str(tmp_path / "up"), id_policy="caller"
    )

    result = await engine.ingest_streams(
        [("a.pdf", io.BytesIO(b"%PDF-a")), ("b.pdf", io.BytesIO(b"%PDF-b"))],
        source_ids=["sid-a", "sid-b"],
    )
    assert result == ["sid-a", "sid-b"]
    assert [j.item.metadata["source_id"] for j in enq.jobs] == ["sid-a", "sid-b"]


async def test_source_ids_length_must_match_items(repo):
    engine, _ = _wire(repo, policy="caller")
    with pytest.raises(ValueError, match="source_ids has 1 entries but 2 documents"):
        await engine.ingest_paths(["/a.txt", "/b.txt"], source_ids=["only-one"])


async def test_caller_policy_requires_ids(repo):
    engine, _ = _wire(repo, policy="caller")
    with pytest.raises(ValueError, match="requires a source_id"):
        await engine.ingest_content([{"content": "x"}])
    with pytest.raises(ValueError, match="requires a source_id"):
        await engine.ingest_paths(["/a.txt"])


async def test_uuid_policy_rejects_supplied_ids(repo):
    engine, _ = _wire(repo, policy="uuid")
    with pytest.raises(ValueError, match="assigns document ids automatically"):
        await engine.ingest_content([{"content": "x", "source_id": "nope"}])
    with pytest.raises(ValueError, match="assigns document ids automatically"):
        await engine.ingest_paths(["/a.txt"], source_ids=["nope"])


async def test_content_hash_enables_dedup_detection(repo):
    """Every document stores a content_hash, queryable independent of the id policy."""
    engine, _ = _wire(repo, auto_drain=True, policy="caller")
    text = "Hello world. " * 10
    h = engine.content_hash(text.encode("utf-8"))

    assert await engine.find_by_content_hash(h) == []  # nothing ingested yet
    await engine.ingest_content([{"content": text, "source_id": "doc-a"}])
    assert await engine.find_by_content_hash(h) == ["doc-a"]

    # Identical content under a different id -> both surface as duplicates.
    await engine.ingest_content([{"content": text, "source_id": "doc-b"}])
    assert sorted(await engine.find_by_content_hash(h)) == ["doc-a", "doc-b"]


async def test_document_id_is_stable_when_content_changes(repo):
    """Re-ingesting under the same id replaces content in place: the id never changes and the
    stored content_hash tracks the new content."""
    engine, _ = _wire(repo, auto_drain=True, policy="caller")
    h1 = engine.content_hash(("alpha " * 20).encode("utf-8"))
    h2 = engine.content_hash(("beta " * 20).encode("utf-8"))

    await engine.ingest_content([{"content": "alpha " * 20, "source_id": "stable"}])
    assert await engine.find_by_content_hash(h1) == ["stable"]

    await engine.ingest_content([{"content": "beta " * 20, "source_id": "stable"}])
    assert (await engine.status("stable")).status == "complete"
    assert await engine.find_by_content_hash(h1) == []          # old content gone
    assert await engine.find_by_content_hash(h2) == ["stable"]  # id stable, hash updated


async def test_content_hash_of_file_matches_what_is_stored(repo, tmp_path):
    engine, _ = _wire(repo, auto_drain=True, policy="caller")
    doc = tmp_path / "note.txt"
    doc.write_text("word " * 40, encoding="utf-8")
    await engine.ingest_paths([str(doc)], source_ids=["f1"])
    assert await engine.find_by_content_hash(engine.content_hash_of_file(str(doc))) == ["f1"]


async def test_delete_removes_document_data_and_status(repo):
    engine, queue = _wire(repo, policy="caller")
    text = "Hello world. " * 10
    await engine.ingest_content([{"content": text, "source_id": "d1"}])
    await queue.run()
    assert (await engine.status("d1")).status == "complete"

    assert await engine.delete_document("d1") is True
    assert await engine.status("d1") is None  # data + job_status both gone
    assert await engine.find_by_content_hash(engine.content_hash(text.encode("utf-8"))) == []


async def test_delete_unknown_document_returns_false(repo):
    engine, _ = _wire(repo)
    assert await engine.delete_document("nope") is False


async def test_delete_clears_pending_jobs(repo):
    """A queued-but-undrained document (job_status only, no data) is still deletable → None."""
    engine, _ = _wire(repo, policy="caller")
    await engine.ingest_content([{"content": "x " * 20, "source_id": "p1"}])  # not drained
    assert (await engine.status("p1")).status == "pending"
    assert await engine.delete_document("p1") is True
    assert await engine.status("p1") is None


async def test_list_documents_inventory(repo):
    engine, queue = _wire(repo, policy="caller")
    await engine.ingest_content([
        {"content": "alpha " * 20, "source_id": "a"},
        {"content": "beta " * 20, "source_id": "b"},
    ])
    await queue.run()

    docs = {d.document_id: d for d in await engine.list_documents()}
    assert set(docs) == {"a", "b"}
    assert docs["a"].chunk_count >= 1
    assert docs["a"].embedding_count == docs["a"].chunk_count
    assert docs["a"].content_hash == engine.content_hash(("alpha " * 20).encode("utf-8"))

    # delete drops it from the inventory.
    assert await engine.delete_document("a") is True
    assert {d.document_id for d in await engine.list_documents()} == {"b"}


async def test_delete_is_atomic_on_failure(repo, monkeypatch):
    """delete() is a single transaction: if any part fails, the whole delete rolls back — the
    document is left fully present (data AND job_status), never half-deleted or a 'ghost' status
    pointing at deleted data. A retry completes the delete."""
    engine, queue = _wire(repo, policy="caller")
    await engine.ingest_content([{"content": "hello world " * 10, "source_id": "d1"}])
    await queue.run()
    assert (await engine.status("d1")).status == "complete"

    # Fail the job-status delete; the data delete ran first in the SAME transaction.
    async def boom(*_args, **_kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(repo, "_delete_job_rows", boom)
    with pytest.raises(RuntimeError, match="delete failed"):
        await engine.delete_document("d1")

    # Atomic rollback: nothing was deleted — data and jobs both intact.
    assert (await engine.status("d1")).status == "complete"
    assert "d1" in {d.document_id for d in await engine.list_documents()}

    # Retry (no fault) completes the delete.
    monkeypatch.undo()
    assert await engine.delete_document("d1") is True
    assert await engine.status("d1") is None


async def test_reingesting_same_source_id_replaces(repo, tmp_path):
    """Re-ingesting under the same source_id upserts (dedup) — chunks are replaced, not appended."""
    engine, _ = _wire(repo, auto_drain=True)
    doc = tmp_path / "note.txt"
    doc.write_text("word " * 40, encoding="utf-8")

    [doc_id] = await engine.ingest_paths([str(doc)], source_ids=["dup-1"])
    assert doc_id == "dup-1"
    first = await engine.status("dup-1")
    assert first.status == "complete" and first.chunk_count >= 1

    # Same id again -> replace, not duplicate.
    await engine.ingest_paths([str(doc)], source_ids=["dup-1"])
    second = await engine.status("dup-1")
    assert second.status == "complete"
    assert second.chunk_count == first.chunk_count  # upsert replaced; no stale/duplicate chunks
    assert second.embedding_count == second.chunk_count
