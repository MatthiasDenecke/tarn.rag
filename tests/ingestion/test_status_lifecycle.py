"""HIGH-02: a re-ingest starts the document's job history afresh — a stale ``failed`` row from an earlier,
superseded run must not pin the document to ``failed`` forever."""

import asyncio
from typing import Literal

import pytest

from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source


class _BrokenExtractor(Extractor):
    """An extractor that always fails — stands in for a corrupt upload."""

    class Config(Extractor.Config):
        class_name: Literal["_broken_extractor"] = "_broken_extractor"

    def extract(self, source: Source):
        raise ValueError("corrupt source")


async def test_a_failed_document_reingested_successfully_reports_complete(tmp_path):
    settings = Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=16,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/status.db"),
    )
    async with await IngestionEngine.create(settings) as engine:
        await engine.ingest_content([{"source_id": "d", "content": "text"}], extractor="_broken_extractor")
        assert (await engine.status("d")).status == "failed"  # the first run dead-lettered

        await engine.ingest_content([{"source_id": "d", "content": "the fixed text"}])
        status = await engine.status("d")
        assert status.status == "complete"  # the stale failed row no longer wins
        assert status.chunk_count == status.embedding_count == 1


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=16,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/status.db"),
    )


async def _state(engine, document_id):
    """(status, stored document text, chunk texts, job statuses) for one document."""
    status = await engine.status(document_id)
    stored = await engine.repository.get_document(document_id)
    chunks = await engine.repository.get_chunks_by_document(document_id)
    jobs = await engine.repository.document_jobs(document_id)
    return status, stored.content, [c.content for c in chunks], {j["status"] for j in jobs}


@pytest.mark.parametrize("first", ["the same text", "an older draft"])
async def test_a_source_id_repeated_in_one_call_is_ingested_once_from_its_last_occurrence(tmp_path, first):
    """Both copies used to run, so their chunk sets shared one batch: identical content collided on the
    content-addressed ids (and dead-lettered), different content stored both sets under one document."""
    async with await IngestionEngine.create(_settings(tmp_path)) as engine:
        ids = await engine.ingest_content([
            {"source_id": "d", "content": first}, {"source_id": "d", "content": "the same text"},
        ])
        status, stored, chunks, jobs = await _state(engine, "d")
        assert ids == ["d", "d"]  # still one id per input
        assert status.status == "complete" and jobs == {"completed"}
        assert stored == "the same text" and chunks == ["the same text"]  # one run: the last copy
        assert status.chunk_count == status.embedding_count == 1
        assert engine._queue.dead_letters == []


async def test_concurrent_embedded_ingests_of_one_document_run_one_after_the_other(tmp_path):
    """Two concurrent calls used to feed the same running waves, so two runs of one document overlapped:
    the older run's Embed job could write after the newer run had replaced the chunk, leaving an orphan
    vector in the dense index. Each embedded submission now runs alone."""
    async with await IngestionEngine.create(_settings(tmp_path)) as engine:
        await asyncio.gather(  # a peer document first, so the second call's root joins the first's waves
            engine.ingest_content([
                {"source_id": "x", "content": "a peer document"}, {"source_id": "d", "content": "version one"},
            ]),
            engine.ingest_content([{"source_id": "d", "content": "version two"}]),
        )
        status, stored, chunks, jobs = await _state(engine, "d")
        assert status.status == "complete" and jobs == {"completed"}
        assert chunks == [stored] and stored in {"version one", "version two"}  # one run's text, not both
        async with engine.repository.engine.connect() as conn:
            vectors = (await conn.exec_driver_sql("SELECT count(*) FROM vec_chunks")).scalar()
        assert vectors == 2  # x's and d's chunk — no orphan from the superseded run


async def test_a_root_job_that_cannot_be_enqueued_reports_failed_not_the_old_state(tmp_path, monkeypatch):
    async with await IngestionEngine.create(_settings(tmp_path)) as engine:
        await engine.ingest_content([{"source_id": "d", "content": "text"}])
        assert (await engine.status("d")).status == "complete"

        async def down(job):
            raise ConnectionError("queue unavailable")

        monkeypatch.setattr(engine.orchestrator.enqueuer, "enqueue", down)
        with pytest.raises(ConnectionError):
            await engine.ingest_content([{"source_id": "d", "content": "new text"}])
        assert (await engine.status("d")).status == "failed"  # the submission failed — say so
        [job] = await engine.repository.document_jobs("d")
        assert job["status"] == "failed" and "queue unavailable" in job["error"]


async def test_a_failed_fan_out_enqueue_leaves_no_phantom_queued_row(tmp_path, monkeypatch):
    """The upstream job is retried (under new downstream job ids), so the job that never reached the queue
    must not keep a ``queued`` row — that would hold the document ``in_progress`` forever."""
    async with await IngestionEngine.create(_settings(tmp_path)) as engine:
        queue = engine.orchestrator.enqueuer
        original, failed = queue.enqueue, []

        async def flaky(job):
            if job.stage_name.startswith("Chunk") and not failed:
                failed.append(job.job_id)
                raise ConnectionError("queue blip")
            await original(job)

        monkeypatch.setattr(queue, "enqueue", flaky)
        await engine.ingest_content([{"source_id": "d", "content": "text"}])
        status, _, chunks, jobs = await _state(engine, "d")
        assert failed and status.status == "complete" and chunks == ["text"]
        assert jobs == {"completed"}
        assert failed[0] not in {j["job_id"] for j in await engine.repository.document_jobs("d")}
