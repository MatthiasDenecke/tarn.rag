"""HIGH-01: section parents (``level > 0``) are stored for the auto-merger, never searched — so a section
and its own leaves can't both land in one result list (the same text twice in the reader's prompt)."""

from tarnrag.contracts import Chunk, ChunkFilter, ChunkProvenance, Document, Embedding
from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.ingestion import IngestionEngine
from tarnrag.retrieval import Query, RetrievalEngine


async def _leaf_and_parent(repo):
    _, (leaf, parent) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s"}),
        [
            Chunk(parent_doc_id="s", content="valve torque check", chunk_index=1,
                  metadata={"parent_ordinal": 0}),
            Chunk(parent_doc_id="s", content="Valves: valve torque check and more", chunk_index=0,
                  provenance=ChunkProvenance(content_hash="p", header_path=["Valves"], level=1)),
        ],
    )
    return leaf, parent


async def test_parents_are_not_sparse_indexed_and_not_returned(repo):
    leaf, parent = await _leaf_and_parent(repo)
    # The parent's vector is what a pre-fix ingest would have written — search must still skip it.
    await repo.store_embeddings([
        Embedding(chunk_id=leaf, vector=[0.0, 0.0, 1.0]),
        Embedding(chunk_id=parent, vector=[0.0, 0.0, 1.0]),
    ])
    async with repo.engine.connect() as conn:
        fts = {r[0] for r in (await conn.exec_driver_sql("SELECT chunk_id FROM fts_chunks")).fetchall()}
    assert fts == {leaf}  # only the retrieval unit is in the sparse index
    for f in (None, ChunkFilter()):
        assert [c.chunk_id for c in await repo.dense_knn([0.0, 0.0, 1.0], 5, f)] == [leaf]
        assert [c.chunk_id for c in await repo.sparse_search("valve torque", 5, f)] == [leaf]
    [record] = await repo.hydrate([parent])
    assert record.provenance.level == 1  # still hydratable for the merger


async def test_a_sectioned_document_indexes_and_returns_only_leaves(tmp_path):
    settings = Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=32,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/units.db"),
    )
    md = (
        "# Pumps\n\nInspect the pump seals monthly.\n\nReplace pump bearings yearly.\n\n"
        "## Valves\n\nCheck valve torque weekly.\n\nLubricate valve stems.\n"
    )
    async with await IngestionEngine.create(settings) as ingest:
        await ingest.ingest_content([{"source_id": "manual", "content": md, "source_type": "md"}])
        chunks = await ingest.repository.get_chunks_by_document("manual")
        status = await ingest.status("manual")
        repo = ingest.repository
        parents = {c.id for c in chunks if c.provenance and c.provenance.level > 0}
        assert parents  # the section tree was built
        assert status.status == "complete"  # every leaf got its vector
        assert status.embedding_count == len(chunks) - len(parents)  # …and no parent did
        retrieval = await RetrievalEngine.open(repo, _embedder(settings), settings)
        hits = await retrieval.search(Query(text="pump valve inspect torque", top_k=20))
        assert hits and not {h.chunk_id for h in hits} & parents


def _embedder(settings):
    from tarnrag.core.resources.embedder import Embedder

    return Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
