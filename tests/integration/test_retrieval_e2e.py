"""End-to-end on one repository: a document through the full DAG, then queried via RetrievalEngine.

Proves the unification — ingestion writes the §8 index and retrieval reads it from the *same*
store (no separate index file). InMemoryJobQueue + SQLite repo + a fake 3-d embedder.
"""

from tarnrag.contracts import IndexMeta, PipelineItem
from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.engine.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.engine.queue import InMemoryJobQueue
from tarnrag.ingestion.engine.result_sink import create_sink_registry
from tarnrag.ingestion.engine.worker import IngestionWorker
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.embed import EmbedStage
from tarnrag.retrieval import Query, RetrievalEngine


class _FakeEmbedder:
    """3-d encoder for both sides: passages by length at ingest, a fixed query vector at search."""

    def embed_passages(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [13.0, 1.0, 0.0]  # ~len of a "Hello world. " chunk

    def config_fingerprint(self):
        return "fp-fake"

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": "fp-fake"}


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


async def test_ingest_then_retrieve_on_one_repo(repo):
    # Ingest the document through the full DAG into the repository...
    queue = InMemoryJobQueue()
    orch = PipelineOrchestrator(PipelineDAG(_stages()), queue, repo, create_sink_registry())
    queue.set_handler(IngestionWorker(orch).handle_batch)
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))  # producer stamps the build record

    await orch.ingest_documents(
        [PipelineItem(content="Hello world. " * 10, metadata={"source_id": "s1"})]
    )
    await queue.run()

    # ...then retrieval reads the SAME repository — no separate index file.
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    results = await engine.search(Query(text="hello world", top_k=3))
    assert results  # the ingested chunks are retrievable
    assert all(r.document_id == "s1" for r in results)
    assert results[0].text  # hydrated text + provenance came back
    assert results[0].license_class == "public_domain"
