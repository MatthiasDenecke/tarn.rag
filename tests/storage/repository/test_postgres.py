"""PostgreSQL dialect (pgvector + tsvector) integration round-trip — gated on a real Postgres.

The whole Postgres path — ``dense_knn`` (pgvector cosine), ``sparse_search`` (tsvector ``ts_rank_cd``),
the Core ``hydrate`` + provenance, the ivfflat / GIN indexes, and FK-cascade delete — is otherwise
unexercised (the default suite is SQLite-only). Set ``TARNRAG_TEST_POSTGRES_URL`` to a pgvector Postgres
to run it, e.g.::

    docker run -d --name pgtest -e POSTGRES_PASSWORD=test -p 5433:5432 pgvector/pgvector:pg16
    TARNRAG_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/postgres \
        pytest tests/storage/repository/test_postgres.py
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import insert

from tarnrag.contracts import Chunk, ChunkFilter, ChunkProvenance, Document, Embedding, MethodRef, Span

PG_URL = os.environ.get("TARNRAG_TEST_POSTGRES_URL", "")
pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.skipif(
        not PG_URL,
        reason="set TARNRAG_TEST_POSTGRES_URL to a pgvector Postgres (e.g. docker pgvector/pgvector)",
    ),
]


@pytest.fixture
async def pg_repo():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from tarnrag.storage.repository.postgres import PostgresRepository

    repo = PostgresRepository(PG_URL, embedding_dimension=3)
    async with repo.engine.begin() as conn:  # clean slate (drops only what already exists)
        await conn.run_sync(repo.metadata.drop_all)
    await repo.connect()  # CREATE EXTENSION vector + tables + ivfflat/GIN indexes
    yield repo
    async with repo.engine.begin() as conn:
        await conn.run_sync(repo.metadata.drop_all)
    await repo.disconnect()


async def _index(repo):
    """Ingest two chunks (+ embeddings): the tank chunk (a) and the quokka chunk (b)."""
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(
                parent_doc_id="s1", content="storage tank corrosion inspection", chunk_index=0,
                total_chunks=2, metadata={"locator": "§6.4"},
                provenance=ChunkProvenance(
                    content_hash="h0", header_path=["Safety"], level=0, geometry=[Span(start=5, end=8)]
                ),
            ),
            Chunk(parent_doc_id="s1", content="quokka marsupial", chunk_index=1, total_chunks=2, metadata={}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=a, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=b, vector=[0.0, 1.0, 0.0]),
    ])
    return a, b


async def test_sparse_quoted_span_is_a_phrase_query(pg_repo):
    """P3 parity: ``websearch_to_tsquery`` turns a double-quoted span into a phrase query — adjacency
    required, so the reversed order does not match."""
    a, _ = await _index(pg_repo)
    assert [c.chunk_id for c in await pg_repo.sparse_search('"tank corrosion"', 5)] == [a]
    assert await pg_repo.sparse_search('"corrosion tank"', 5) == []  # reversed ⇒ not a phrase


async def test_store_embeddings_upserts_on_chunk_id(pg_repo):
    """Schema v2: ``chunk_id`` is the PK, so re-embedding a chunk REPLACES its vector in place
    (``ON CONFLICT`` — the Postgres counterpart of SQLite's ``INSERT OR REPLACE``); a second row per
    chunk is impossible by construction. Empty input is a no-op."""
    a, _ = await _index(pg_repo)
    assert await pg_repo.store_embeddings([]) == []  # the empty early-out
    # Re-embed chunk a pointing the other way; the [0,0,1] query must now rank it first — one row, updated.
    assert await pg_repo.store_embeddings([Embedding(chunk_id=a, vector=[0.0, 0.0, 1.0])]) == [a]
    hits = await pg_repo.dense_knn([0.0, 0.0, 1.0], 5)
    assert hits[0].chunk_id == a
    assert [c.chunk_id for c in hits].count(a) == 1  # replaced, not duplicated


async def test_dense_sparse_hydrate_roundtrip(pg_repo):
    a, b = await _index(pg_repo)
    # pgvector cosine KNN: the [1,0,0] query is nearest the tank chunk.
    assert [c.chunk_id for c in await pg_repo.dense_knn([1.0, 0.0, 0.0], 2)][0] == a
    # tsvector ts_rank_cd: "tank inspection" surfaces the tank chunk.
    assert [c.chunk_id for c in await pg_repo.sparse_search("tank inspection", 5)][0] == a
    # hydrate carries licensing + layout-aware provenance (the shared Core _chunk_provenance path).
    recs = {r.chunk_id: r for r in await pg_repo.hydrate([a, b])}
    assert recs[a].locator == "§6.4" and recs[a].ai_grounding_allowed is True
    assert recs[a].provenance.header_path == ["Safety"] and recs[a].provenance.level == 0
    assert recs[a].provenance.geometry[0].start == 5


async def test_section_parents_are_hydrated_but_never_searched(pg_repo):
    """HIGH-01: a section parent (level > 0) is stored for the auto-merger to hydrate, but dense + sparse
    search return retrieval units only — even from an index built when parents were still embedded."""
    _, (leaf, parent) = await pg_repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s2"}),
        [
            Chunk(parent_doc_id="s2", content="valve torque check", chunk_index=1,
                  metadata={"parent_ordinal": 0}),
            Chunk(parent_doc_id="s2", content="Valves: valve torque check and more", chunk_index=0,
                  provenance=ChunkProvenance(content_hash="p", header_path=["Valves"], level=1)),
        ],
    )
    await pg_repo.store_embeddings([  # the parent's vector is what a pre-fix ingest would have written
        Embedding(chunk_id=leaf, vector=[0.0, 0.0, 1.0]),
        Embedding(chunk_id=parent, vector=[0.0, 0.0, 1.0]),
    ])
    assert [c.chunk_id for c in await pg_repo.dense_knn([0.0, 0.0, 1.0], 5)] == [leaf]
    assert [c.chunk_id for c in await pg_repo.dense_knn([0.0, 0.0, 1.0], 5, ChunkFilter())] == [leaf]
    assert [c.chunk_id for c in await pg_repo.sparse_search("valve torque", 5)] == [leaf]
    [record] = await pg_repo.hydrate([parent])
    assert record.provenance.level == 1  # still hydratable for the merger


async def test_filter_drops_disallowed_and_scopes(pg_repo):
    """The permitted-chunk filter (available / method scope) is applied inside dense + sparse with
    over-fetch backfill — the Postgres side of finding 1.2."""
    a, b = await _index(pg_repo)  # a = tank chunk, b = quokka chunk
    async with pg_repo.engine.begin() as conn:
        await conn.execute(pg_repo.chunks.update().where(pg_repo.chunks.c.chunk_id == b).values(available=0))
        await conn.execute(insert(pg_repo.method_chunks), [{"method_id": "M1", "method_version": "v1", "chunk_id": a}])
    # [0,1,0] is nearest the (now unavailable) quokka chunk -> available-only backfills to the tank chunk.
    avail = await pg_repo.dense_knn([0.0, 1.0, 0.0], 5, ChunkFilter(require_available=True))
    assert [c.chunk_id for c in avail] == [a]
    # method scope restricts dense + sparse to the in-scope chunk only.
    only_a = ChunkFilter(method_scope=(MethodRef("M1"),))
    assert [c.chunk_id for c in await pg_repo.dense_knn([1.0, 0.0, 0.0], 5, only_a)] == [a]
    assert [c.chunk_id for c in await pg_repo.sparse_search("tank", 5, only_a)] == [a]  # a in scope
    assert await pg_repo.sparse_search("quokka", 5, only_a) == []  # b matches but is out of scope
    # license-class axis: neither chunk is customer_licensed (both default public_domain) -> nothing permitted.
    assert await pg_repo.dense_knn([1.0, 0.0, 0.0], 5, ChunkFilter(license_classes=("customer_licensed",))) == []


async def test_delete_cascades(pg_repo):
    a, _ = await _index(pg_repo)
    assert (await pg_repo.document_facts("s1")).present
    assert await pg_repo.delete_document_and_jobs("s1") is True
    assert await pg_repo.hydrate([a]) == []  # chunks + embeddings cascaded off the FKs
    assert not (await pg_repo.document_facts("s1")).present


async def test_reingest_replaces_chunks(pg_repo):
    await _index(pg_repo)  # two chunks
    # Re-store the same source_id with one chunk -> idempotent replace (not a duplicate).
    await pg_repo.store_document_with_chunks(
        Document(content="d2", metadata={"source_id": "s1"}),
        [Chunk(parent_doc_id="s1", content="only one now", chunk_index=0, total_chunks=1, metadata={})],
    )
    assert (await pg_repo.document_facts("s1")).chunk_count == 1


async def test_concurrent_upserts_of_one_key_never_collide(pg_repo):
    """HIGH-04: the old UPDATE-then-INSERT let two READ COMMITTED writers both miss the UPDATE and collide
    on the INSERT (IntegrityError). ``INSERT … ON CONFLICT DO UPDATE`` is one atomic statement."""
    import asyncio

    for trial in range(5):
        meta = [pg_repo.write_index_meta({f"k{trial}": str(i)}) for i in range(8)]
        docs = [
            pg_repo.store_document(Document(content=str(i), metadata={"source_id": f"race{trial}"}))
            for i in range(8)
        ]
        results = await asyncio.gather(*meta, *docs, return_exceptions=True)
        assert not [r for r in results if isinstance(r, Exception)]
    assert len(await pg_repo.index_meta()) == 5
