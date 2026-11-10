"""End-to-end: a document through the whole pipeline on InMemoryJobQueue + SQLite.

No Postgres, no pgQueuer, no embedding model — the embed stage uses a fake encoder.
"""

from sqlalchemy import func, select

from tarnrag.contracts import PipelineItem
from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue
from tarnrag.ingestion.engine.result_sink import create_sink_registry
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.embed import EmbedStage


class _FakeEmbedder:
    """Deterministic 3-d encoder — no sentence-transformers needed."""

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


async def test_full_document_ingest(repo):
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(PipelineDAG(_stages()), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)  # pure handler; gets stages from the DAG
    queue.set_handler(worker.handle_batch)  # composition-root wiring

    item = PipelineItem(content="Hello world. " * 10, metadata={"source_id": "s1"})
    [doc_id] = await orch.ingest_documents([item])
    assert doc_id == "s1"

    await queue.run()  # drain the whole DAG, including chunk fan-out

    # Document fully ingested: doc + N chunks + N embeddings, no failures.
    status = await repo.document_status("s1")
    assert status["status"] == "complete"
    assert status["chunk_count"] >= 2
    assert status["embedding_count"] == status["chunk_count"]
    assert queue.dead_letters == []

    # Embeddings are searchable; chunks hydrate with their §8 provenance.
    cands = await repo.dense_knn([10.0, 1.0, 0.0], k=status["chunk_count"])
    assert len(cands) == status["chunk_count"]
    recs = await repo.hydrate([c.chunk_id for c in cands])
    assert all(r.text and r.license_class for r in recs)

    # Every job for the document ended 'completed' (none failed/stuck).
    jobs = await repo.document_jobs("s1")
    assert jobs and all(j["status"] == "completed" for j in jobs)


async def test_reingest_is_idempotent(repo):
    """Re-ingesting the same source_id replaces chunks/embeddings, not duplicates."""
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(PipelineDAG(_stages()), queue, repo, create_sink_registry())
    worker = IngestionWorker(orch)
    queue.set_handler(worker.handle_batch)

    item = PipelineItem(content="alpha beta gamma. " * 8, metadata={"source_id": "s1"})
    await orch.ingest_documents([item])
    await queue.run()
    first = await repo.document_status("s1")

    await orch.ingest_documents(
        [PipelineItem(content="alpha beta gamma. " * 8, metadata={"source_id": "s1"})]
    )
    await queue.run()
    second = await repo.document_status("s1")

    assert second["status"] == "complete"
    # Counts didn't double — the document was upserted and its chunks replaced.
    assert second["chunk_count"] == first["chunk_count"]
    assert second["embedding_count"] == first["embedding_count"]

    # Prove no orphans: chunks (FK CASCADE) and vec_chunks (explicit clear — vec0 doesn't
    # cascade) hold exactly the live rows after re-ingest.
    async with repo.engine.connect() as conn:
        raw_chunks = (await conn.execute(select(func.count()).select_from(repo.chunks))).scalar_one()
        raw_vecs = (await conn.exec_driver_sql("SELECT count(*) FROM vec_chunks")).scalar_one()
    assert raw_chunks == second["chunk_count"]
    assert raw_vecs == second["embedding_count"]
