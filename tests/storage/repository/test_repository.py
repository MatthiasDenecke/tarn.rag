import hashlib

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from tarnrag.contracts import (
    Annotation,
    Chunk,
    ChunkFilter,
    ChunkProvenance,
    Document,
    Embedding,
    MethodRef,
    PageBox,
    Span,
    Table,
    TableCell,
)
from tarnrag.storage.repository.sqlite import SqliteRepository


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(content, index, total, source_id="s1", **meta):
    return Chunk(
        parent_doc_id="",  # overwritten by store_document_with_chunks
        content=content,
        chunk_index=index,
        total_chunks=total,
        metadata={"source_id": source_id, **meta},
    )


async def test_connect_creates_missing_parent_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    repo = SqliteRepository(f"sqlite:///{nested}/docs.db", embedding_dimension=3)
    await repo.connect()
    assert nested.is_dir()  # SQLite won't create it; connect() does
    await repo.disconnect()


async def test_store_and_get_document(repo):
    doc_id = await repo.store_document(
        Document(content="hello", metadata={"source_id": "s1"})
    )
    got = await repo.get_document(doc_id)
    assert got is not None
    assert got.content == "hello"
    assert got.metadata["source_id"] == "s1"


async def test_corpus_stats_summarizes_counts_and_document_lengths(repo):
    await repo.store_document_with_chunks(
        Document(content="x" * 10, metadata={"source_id": "a"}),
        [_chunk("c0", 0, 2, source_id="a"), _chunk("c1", 1, 2, source_id="a")],
    )
    await repo.store_document_with_chunks(
        Document(content="y" * 30, metadata={"source_id": "b"}),
        [_chunk("only", 0, 1, source_id="b")],
    )
    stats = await repo.corpus_stats()
    assert stats.document_count == 2
    assert stats.chunk_count == 3
    assert stats.embedding_count == 0  # none stored
    assert (stats.min_chars, stats.max_chars, stats.total_chars) == (10, 30, 40)
    assert stats.mean_chars == 20.0 and stats.median_chars == 20.0  # median of [10, 30]
    assert stats.mean_chunks_per_doc == 1.5  # 3 chunks / 2 docs


async def test_corpus_stats_on_an_empty_repository_is_all_zeros(repo):
    stats = await repo.corpus_stats()
    assert stats.document_count == 0 and stats.chunk_count == 0 and stats.total_chars == 0
    assert stats.mean_chars == 0.0 and stats.median_chars == 0.0 and stats.mean_chunks_per_doc == 0.0


async def test_document_with_chunks_and_idempotency(repo):
    doc_id, ids = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}),
        [_chunk("c0", 0, 2), _chunk("c1", 1, 2)],
    )
    assert len(ids) == 2
    assert len(await repo.get_chunks_by_document(doc_id)) == 2

    # Re-ingesting the same source_id upserts the document (same id) and replaces chunks.
    doc_id2, _ = await repo.store_document_with_chunks(
        Document(content="updated", metadata={"source_id": "s1"}),
        [_chunk("only", 0, 1)],
    )
    assert doc_id2 == doc_id
    remaining = await repo.get_chunks_by_document(doc_id)
    assert [c.content for c in remaining] == ["only"]
    assert (await repo.get_document(doc_id)).content == "updated"


async def test_chunk_provenance_round_trips(repo):
    chunk = Chunk(
        parent_doc_id="",
        content="Wear PPE.",
        chunk_index=0,
        provenance=ChunkProvenance(
            header_path=["Manual", "Safety"],
            geometry=[Span(start=39, end=48, boxes=[PageBox(page=1, bbox=(10, 20, 200, 35))])],
            content_hash=_h("Wear PPE."),
            level=0,
        ),
        metadata={"source_id": "s1"},
    )
    doc_id, _ = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}), [chunk]
    )
    [got] = await repo.get_chunks_by_document(doc_id)
    assert got.provenance is not None
    assert got.provenance.header_path == ["Manual", "Safety"]
    assert got.provenance.content_hash == _h("Wear PPE.")
    span = got.provenance.geometry[0]
    assert (span.start, span.end) == (39, 48)
    assert span.boxes[0].page == 1 and span.boxes[0].bbox == (10.0, 20.0, 200.0, 35.0)  # PDF highlight geometry


async def test_parent_chunk_id_resolves_from_parent_ordinal(repo):
    def c(content, idx, level, parent_ordinal):
        return Chunk(
            parent_doc_id="",
            content=content,
            chunk_index=idx,
            provenance=ChunkProvenance(content_hash=_h(content), level=level, header_path=["S"]),
            metadata={"source_id": "s1", "parent_ordinal": parent_ordinal},
        )

    doc_id, _ = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}),
        [c("Safety section", 0, 1, None), c("Wear PPE.", 1, 0, 0), c("Lock out.", 2, 0, 0)],
    )
    parent, leaf1, leaf2 = await repo.get_chunks_by_document(doc_id)  # ordered by ordinal
    assert parent.provenance.level == 1 and parent.provenance.parent_chunk_id is None
    assert leaf1.provenance.parent_chunk_id == parent.id  # the positional ordinal -> the parent's chunk_id
    assert leaf2.provenance.parent_chunk_id == parent.id


async def test_table_chunk_persists_cells_and_round_trips(repo):
    table = Table(
        n_rows=2,
        n_cols=2,
        markdown="| Bolt | Torque |",
        cells=[
            TableCell(id="c0", row=0, col=0, is_column_header=True, text="Bolt", geometry=[Span(start=0, end=4)]),
            TableCell(id="c1", row=0, col=1, is_column_header=True, text="Torque"),
            TableCell(id="c2", row=1, col=0, is_row_header=True, text="M6"),
            TableCell(id="c3", row=1, col=1, text="6 Nm"),
        ],
    )
    chunk = Chunk(
        parent_doc_id="",
        content="| Bolt | Torque |\n| M6 | 6 Nm |",
        chunk_index=0,
        provenance=ChunkProvenance(content_hash=_h("table"), header_path=["Specs"], table=table),
        metadata={"source_id": "s1"},
    )
    doc_id, _ = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}), [chunk]
    )
    [got] = await repo.get_chunks_by_document(doc_id)
    t = got.provenance.table
    assert t is not None and (t.n_rows, t.n_cols) == (2, 2)  # derived from the persisted cells
    data = t.cell_at(1, 1)
    assert data.text == "6 Nm"
    assert [c.text for c in t.column_headers_for(data)] == ["Torque"]  # query by column header
    assert [c.text for c in t.row_headers_for(data)] == ["M6"]  # query by row header
    assert t.cell_at(0, 0).geometry[0].start == 0  # cell geometry round-trips


async def test_chunk_annotations_round_trip(repo):
    chunk = Chunk(
        parent_doc_id="",
        content="Wear PPE.",
        chunk_index=0,
        provenance=ChunkProvenance(
            content_hash=_h("Wear PPE."),
            annotations=[
                Annotation(
                    producer="acronyms", type="entity", value={"text": "PPE"},
                    span=[Span(start=5, end=8)], deterministic=True,
                ),
                Annotation(producer="topics", type="topic", value={"label": "safety"}, deterministic=False),
            ],
        ),
        metadata={"source_id": "s1"},
    )
    doc_id, _ = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}), [chunk]
    )
    [got] = await repo.get_chunks_by_document(doc_id)
    entity, topic = got.provenance.annotations  # position-ordered
    assert (entity.producer, entity.type, entity.value) == ("acronyms", "entity", {"text": "PPE"})
    assert (entity.span[0].start, entity.span[0].end) == (5, 8) and entity.deterministic is True
    assert topic.type == "topic" and topic.span is None and topic.deterministic is False  # doc-level, generative


async def test_update_chunk_metadata_is_noop(repo):
    # §8 chunks carry no metadata column — update_chunk_metadata is a harmless no-op
    # (enrichment is not persisted; the metadata bag is deferred).
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1)],
    )
    await repo.update_chunk_metadata(cid, {"k": 2})       # does not raise, stores nothing
    await repo.update_chunk_metadata("missing", {"k": 2})  # also a no-op for unknown ids
    assert "k" not in (await repo.get_chunk(cid)).metadata


async def test_document_status_and_jobs(repo):
    assert await repo.document_status("unknown") is None

    # A queued job with no document yet -> pending.
    await repo.record_job("s1", "j1", "LoadAndParse", "queued")
    assert (await repo.document_status("s1"))["status"] == "pending"

    # Document + chunk + embedding -> complete.
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1)],
    )
    await repo.store_embeddings(
        [Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0])]
    )
    # The data is all there, but its job is still queued — e.g. a re-ingest whose new run hasn't
    # replaced the old data yet: in progress, not complete.
    assert (await repo.document_status("s1"))["status"] == "in_progress"
    await repo.record_job("s1", "j1", "LoadAndParse", "completed")
    assert await repo.document_status("s1") == {
        "document_id": "s1",
        "status": "complete",
        "chunk_count": 1,
        "embedding_count": 1,
    }

    # A failed job rolls the document status up to 'failed'.
    await repo.record_job("s1", "j2", "Embed", "failed", "boom")
    assert (await repo.document_status("s1"))["status"] == "failed"

    jobs = await repo.document_jobs("s1")
    assert len(jobs) == 2
    assert any(j["status"] == "failed" and j["error"] == "boom" for j in jobs)


async def test_delete_document_keeps_jobs_until_cleared(repo):
    # The granular ports: delete_document removes only the data; job_status survives until
    # delete_document_jobs clears it. (delete_document_and_jobs does both in one shot.)
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1)],
    )
    await repo.store_embeddings(
        [Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0])]
    )
    await repo.record_job("s1", "j1", "LoadAndParse", "completed")
    assert (await repo.document_status("s1"))["status"] == "complete"

    # delete_document drops the data (chunks/embeddings cascade); job_status survives until cleared.
    assert await repo.delete_document("s1") is True
    facts = await repo.document_facts("s1")
    assert facts.present is False and facts.chunk_count == 0 and facts.embedding_count == 0
    assert (await repo.document_status("s1"))["status"] == "pending"  # stale job row remains

    assert await repo.delete_document_jobs("s1") is True
    assert await repo.document_status("s1") is None  # fully gone

    # Idempotent: nothing left to remove.
    assert await repo.delete_document("s1") is False
    assert await repo.delete_document_jobs("s1") is False


async def test_delete_document_and_jobs_removes_both(repo):
    # The atomic combined delete: data + job_status gone in a single transaction.
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("a", 0, 1)],
    )
    await repo.store_embeddings(
        [Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0])]
    )
    await repo.record_job("s1", "j1", "LoadAndParse", "completed")
    assert (await repo.document_status("s1"))["status"] == "complete"

    assert await repo.delete_document_and_jobs("s1") is True
    assert await repo.document_status("s1") is None  # data AND jobs gone in one call
    assert await repo.delete_document_and_jobs("s1") is False  # idempotent


async def test_list_documents(repo):
    await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1", "content_hash": "h1"}),
        [_chunk("a", 0, 2), _chunk("b", 1, 2)],
    )
    await repo.store_document(  # second doc, no chunks
        Document(content="e", metadata={"source_id": "s2", "content_hash": "h2"})
    )

    docs = {d["document_id"]: d for d in await repo.list_documents()}
    assert set(docs) == {"s1", "s2"}
    assert docs["s1"]["content_hash"] == "h1"
    assert docs["s1"]["chunk_count"] == 2 and docs["s1"]["embedding_count"] == 0
    assert docs["s2"]["chunk_count"] == 0 and docs["s2"]["content_hash"] == "h2"


async def test_store_document_is_atomic(repo, monkeypatch):
    # Existing document with content "A".
    doc_id = await repo.store_document(Document(content="A", metadata={"source_id": "s1"}))

    # Force the chunk-delete (the second statement) to fail mid-transaction.
    def boom(*args, **kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(repo.chunks, "delete", boom)
    with pytest.raises(RuntimeError, match="delete failed"):
        await repo.store_document(Document(content="B", metadata={"source_id": "s1"}))

    # The upsert to "B" shared the same transaction, so it was rolled back: still "A".
    assert (await repo.get_document(doc_id)).content == "A"


async def test_refresh_documents_rewrites_the_row_and_keeps_chunks_vectors_and_bm25(repo):
    """``refresh_documents`` (the CleanAndNormalize sink's write) is row-only — unlike ``store_documents``,
    it never replaces the document's chunks, so a replayed job can't wipe the index built after it."""
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="raw  text", metadata={"source_id": "s1", "content_hash": "h1"}),
        [_chunk("tank inspection", 0, 1)],
    )
    await repo.store_embeddings([Embedding(chunk_id=cid, vector=[1.0, 0.0, 0.0])])
    ids = await repo.refresh_documents(
        [Document(content="raw text", metadata={"source_id": "s1", "content_hash": "h1"})]
    )
    assert ids == ["s1"] and (await repo.get_document("s1")).content == "raw text"
    facts = await repo.document_facts("s1")
    assert (facts.chunk_count, facts.embedding_count) == (1, 1)
    assert [c.chunk_id for c in await repo.dense_knn([1.0, 0.0, 0.0], k=1)] == [cid]
    assert [c.chunk_id for c in await repo.sparse_search("tank", k=1)] == [cid]


async def test_dense_knn_and_hydrate(repo):
    _, (cid_a, cid_b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("tank inspection", 0, 2), _chunk("quokka", 1, 2)],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cid_a, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=cid_b, vector=[0.0, 1.0, 0.0]),
    ])
    cands = await repo.dense_knn([0.9, 0.1, 0.0], k=2)
    assert cands[0].chunk_id == cid_a  # nearest first
    recs = await repo.hydrate([cid_a])
    assert recs[0].text == "tank inspection" and recs[0].document_id == "s1"


async def test_fts_index_is_populated(repo):
    await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("storage tank inspection", 0, 1)],
    )
    async with repo.engine.connect() as conn:
        rows = (
            await conn.exec_driver_sql("SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH 'tank'")
        ).fetchall()
    assert len(rows) == 1


async def test_sparse_search_ranks_lexical_matches(repo):
    await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("storage tank corrosion inspection", 0, 2), _chunk("quokka marsupial habitat", 1, 2)],
    )
    hits = await repo.sparse_search("tank corrosion", k=5)
    assert hits and hits[0].rank == 1
    [rec] = await repo.hydrate([hits[0].chunk_id])
    assert rec.text == "storage tank corrosion inspection"  # the lexical match ranks first
    assert await repo.sparse_search("zzznomatch", k=5) == []  # no terms match -> empty


async def test_dense_knn_filter_backfills_past_disallowed(repo):
    """A filtered dense KNN returns the k nearest *permitted* chunks — over-fetch backfills past the
    nearest chunks the filter drops, so a tight filter can't under-return (finding 1.2)."""
    _, (c1, c2, c3, c4) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            _chunk("nearest A", 0, 4, available=0),  # nearest, but unavailable
            _chunk("nearest B", 1, 4, ai_grounding_allowed=0),  # 2nd nearest, not grounding-allowed
            _chunk("third C", 2, 4),  # permitted
            _chunk("far D", 3, 4),  # permitted
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=c1, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=c2, vector=[0.9, 0.1, 0.0]),
        Embedding(chunk_id=c3, vector=[0.8, 0.2, 0.0]),
        Embedding(chunk_id=c4, vector=[0.0, 1.0, 0.0]),
    ])
    query = [1.0, 0.0, 0.0]  # nearest order: c1, c2, c3, c4
    assert [c.chunk_id for c in await repo.dense_knn(query, k=2)] == [c1, c2]  # unfiltered: nearest two
    # available-only: c1 (unavailable) drops; the next permitted backfills.
    avail = await repo.dense_knn(query, k=2, filter=ChunkFilter(require_available=True))
    assert [c.chunk_id for c in avail] == [c2, c3]
    # also require grounding: c1 (unavailable) AND c2 (not grounding) drop -> c3, c4.
    grounded = await repo.dense_knn(
        query, k=2, filter=ChunkFilter(require_available=True, require_grounding=True)
    )
    assert [c.chunk_id for c in grounded] == [c3, c4]
    assert [c.rank for c in grounded] == [1, 2]  # re-ranked 1..k over the permitted set


async def test_dense_knn_filter_restricts_to_method_scope(repo):
    """A method-scoped filter restricts dense results to chunks reachable from the scoped methods (the
    SQL method_chunks pre-filter); an empty scope permits nothing."""
    _, (c1, c2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("in scope", 0, 2), _chunk("out of scope", 1, 2)],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=c1, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=c2, vector=[0.9, 0.1, 0.0]),
    ])
    async with repo.engine.begin() as conn:
        await conn.execute(
            insert(repo.method_chunks),
            [{"method_id": "M1", "method_version": "v1", "chunk_id": c1}],
        )
    scoped = await repo.dense_knn([1.0, 0.0, 0.0], k=5, filter=ChunkFilter(method_scope=(MethodRef("M1"),)))
    assert [c.chunk_id for c in scoped] == [c1]  # version-agnostic MethodRef matches v1
    assert await repo.dense_knn([1.0, 0.0, 0.0], k=5, filter=ChunkFilter(method_scope=())) == []


async def test_method_bundle_registers_replaces_and_clears(repo):
    """``register_method_bundle`` is the §8 bundle writer the scope-filter path was missing: it writes
    the bundle, REPLACES it on re-registration (idempotent), clears on an empty list — and a scoped
    dense search honors it end to end."""
    _, (c1, c2, c3) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("one", 0, 3), _chunk("two", 1, 3), _chunk("three", 2, 3)],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=c, vector=v)
        for c, v in ((c1, [1.0, 0.0, 0.0]), (c2, [0.9, 0.1, 0.0]), (c3, [0.8, 0.2, 0.0]))
    ])
    await repo.register_method_bundle("M1", "v1", [c1, c2])
    assert await repo.method_bundle("M1", "v1") == sorted([c1, c2])
    scoped = await repo.dense_knn(
        [1.0, 0.0, 0.0], k=5, filter=ChunkFilter(method_scope=(MethodRef("M1", "v1"),))
    )
    assert {c.chunk_id for c in scoped} == {c1, c2}  # the registered bundle, through the filter

    await repo.register_method_bundle("M1", "v1", [c3])  # re-register REPLACES the bundle
    assert await repo.method_bundle("M1", "v1") == [c3]
    await repo.register_method_bundle("M1", "v1", [])  # empty clears it
    assert await repo.method_bundle("M1", "v1") == []
    assert (
        await repo.dense_knn([1.0, 0.0, 0.0], k=5, filter=ChunkFilter(method_scope=(MethodRef("M1"),)))
        == []
    )


async def test_hydrate_returns_registered_methods_batched(repo):
    """``hydrate`` carries each chunk's method refs (now fetched in ONE query, not one per chunk),
    ordered by method id for determinism."""
    _, (c1, c2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("one", 0, 2), _chunk("two", 1, 2)],
    )
    await repo.register_method_bundle("M2", "v2", [c1, c2])
    await repo.register_method_bundle("M1", "v1", [c1])
    rec1, rec2 = await repo.hydrate(["not-a-chunk", c1, c2])  # unknown ids are skipped, order kept
    assert rec1.methods == [("M1", "v1"), ("M2", "v2")]  # ordered by method_id
    assert rec2.methods == [("M2", "v2")]
    async with repo.engine.connect() as conn:  # the helpers' empty early-outs (hydrate returns before them)
        assert await repo.reads.methods_by_chunk(conn, []) == {}
        assert await repo.reads.chunk_provenance(conn, []) == {}
        assert await repo.reads.child_rows_by_chunk(conn, repo.table_cells, []) == {}


def test_fts_query_composes_exact_match_intent():
    """The FTS5 MATCH builder (P3): plain queries are byte-identical to the old OR builder; quoted spans
    and punctuation-split identifiers become REQUIRED phrases, AND-ed with the OR remainder."""
    q = SqliteRepository._fts_query
    assert q("tank inspection") == '"tank" OR "inspection"'  # plain: unchanged
    assert q('check the "goodwill impairment" note') == '"goodwill impairment" AND ("check" OR "the" OR "note")'
    assert q("torque for 6.4.2") == '"6 4 2" AND ("torque" OR "for")'  # dotted identifier ⇒ adjacency
    assert q("§6.4 applies") == '"6 4" AND ("applies")'
    assert q('"exact phrase"') == '"exact phrase"'  # phrase-only query
    assert q("BM25 ranking") == '"bm25" OR "ranking"'  # single-piece identifier is a plain token already
    assert q('find " " it') == '"find" OR "it"'  # a blank quoted span contributes nothing
    assert q("under 20,000,000 total") == '"under" OR "20" OR "000" OR "total"'  # dedupe within one token too
    assert q("!!! ...") == ""  # nothing usable ⇒ no match


async def test_sparse_search_requires_phrase_adjacency(repo):
    """A quoted span or dotted identifier must appear as an ADJACENT phrase — token scatter no longer
    matches (P3); plain multi-token queries stay recall-first (OR)."""
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            _chunk("apply the torque from section 6.4.2 of the manual", 0, 2),
            _chunk("6 bolts, 4 nuts and 2 washers in the torque kit", 1, 2),
        ],
    )
    plain = await repo.sparse_search("torque kit", 5)  # plain OR: both chunks match
    assert {c.chunk_id for c in plain} == {a, b}
    scoped = await repo.sparse_search("torque 6.4.2", 5)  # the identifier is a required phrase
    assert [c.chunk_id for c in scoped] == [a]  # b has 6 / 4 / 2 scattered — excluded
    quoted = await repo.sparse_search('"torque kit"', 5)  # quoted span: adjacency required
    assert [c.chunk_id for c in quoted] == [b]  # a has 'torque from …' — not the phrase
    assert await repo.sparse_search("!!! ...", 5) == []  # nothing usable ⇒ no match, no FTS error


async def test_hydrate_carries_table_and_annotations(repo):
    """``hydrate`` (the retrieval read path) rebuilds a chunk's table cells AND annotations onto its
    provenance — the same assembly ``get_chunk`` uses, through the hydrate-side ``chunk_provenance``."""
    chunk = Chunk(
        parent_doc_id="",
        content="| Bolt | Torque |",
        chunk_index=0,
        provenance=ChunkProvenance(
            content_hash=_h("t"),
            table=Table(n_rows=1, n_cols=1, cells=[TableCell(id="c0", row=0, col=0, text="Bolt")]),
            annotations=[Annotation(producer="acronyms", type="entity", value={"text": "M6"})],
        ),
        metadata={"source_id": "s1"},
    )
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="full", metadata={"source_id": "s1"}), [chunk]
    )
    [rec] = await repo.hydrate([cid])
    assert rec.provenance.table is not None and rec.provenance.table.cell_at(0, 0).text == "Bolt"
    assert [a.type for a in rec.provenance.annotations] == ["entity"]


async def test_dense_knn_filter_by_license_class(repo):
    """The permitted-chunk filter restricts by license_class (the ModusQ §5.6 policy axis)."""
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("open", 0, 2, license_class="public_domain"),
         _chunk("copyrighted", 1, 2, license_class="third_party_copyrighted")],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=a, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=b, vector=[0.9, 0.1, 0.0]),
    ])
    hits = await repo.dense_knn(
        [1.0, 0.0, 0.0], k=5, filter=ChunkFilter(license_classes=("public_domain", "customer_licensed"))
    )
    assert [c.chunk_id for c in hits] == [a]  # the copyrighted chunk is filtered out


async def test_sparse_search_filter_drops_unavailable(repo):
    """The permitted-chunk filter applies to sparse retrieval too."""
    _, (_c1, c2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [_chunk("tank inspection alpha", 0, 2, available=0), _chunk("tank inspection beta", 1, 2)],
    )
    hits = await repo.sparse_search("tank inspection", k=5, filter=ChunkFilter(require_available=True))
    assert [h.chunk_id for h in hits] == [c2]  # the unavailable lexical match is dropped


async def test_hydrate_carries_layout_aware_provenance(repo):
    _, (cid,) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(
                parent_doc_id="", content="Wear PPE.", chunk_index=0,
                provenance=ChunkProvenance(
                    content_hash=_h("Wear PPE."), header_path=["Safety"], level=1,
                    geometry=[Span(start=5, end=8)],
                ),
                metadata={"source_id": "s1"},
            )
        ],
    )
    [rec] = await repo.hydrate([cid])
    assert rec.provenance is not None
    assert (rec.provenance.header_path, rec.provenance.level) == (["Safety"], 1)
    assert rec.provenance.geometry[0].start == 5


async def test_license_class_enum_is_enforced(repo):
    # license_class is a closed §8 enum guarded by a CHECK constraint.
    with pytest.raises(IntegrityError):
        await repo.store_document(
            Document(content="d", metadata={"source_id": "s1", "license_class": "bogus"})
        )


async def test_index_meta_round_trip(repo):
    assert await repo.index_meta() == {}  # empty before anything is written
    await repo.write_index_meta({"schema_version": "1", "embedding_config_fingerprint": "fp-1"})
    assert await repo.index_meta() == {
        "schema_version": "1", "embedding_config_fingerprint": "fp-1",
    }
    # Upsert: re-writing a key updates it in place (no duplicate row).
    await repo.write_index_meta({"schema_version": "2"})
    meta = await repo.index_meta()
    assert meta["schema_version"] == "2" and meta["embedding_config_fingerprint"] == "fp-1"


async def test_store_chunks_is_idempotent_on_retry(repo):
    """HIGH-03: a retried Chunk job (worker died after the commit, before the ack) rewrites the same chunk
    set — same content-addressed ids, no second copy in ``chunks`` or the sparse index."""
    await repo.store_document(Document(content="d", metadata={"source_id": "s1"}))

    def batch(first_text: str = "alpha") -> list:  # what ChunkResultSink builds: parent_doc_id is set
        return [
            _chunk(text, i, 2).model_copy(update={"parent_doc_id": "s1"})
            for i, text in enumerate([first_text, "beta"])
        ]

    first = await repo.store_chunks(batch())
    second = await repo.store_chunks(batch())
    assert first == second  # deterministic ids — in-flight Embed jobs still point at real chunks
    assert (await repo.document_facts("s1")).chunk_count == 2
    async with repo.engine.connect() as conn:
        fts = (await conn.exec_driver_sql("SELECT count(*) FROM fts_chunks")).fetchone()[0]
    assert fts == 2
    # Changed content at the same position is a different chunk (a stale citation can't resolve to it).
    [changed, _] = await repo.store_chunks(batch("alpha v2"))
    assert changed != first[0] and await repo.get_chunk(first[0]) is None


async def test_upsert_inserts_then_updates_in_place(repo):
    await repo.write_index_meta({"k": "1"})
    await repo.write_index_meta({"k": "2", "j": "x"})
    assert await repo.index_meta() == {"k": "2", "j": "x"}
    await repo.record_job("s1", "j1", "Chunk", "queued")
    await repo.record_job("s1", "j1", "Chunk", "failed", "boom")  # same job: updated, not duplicated
    assert await repo.document_jobs("s1") == [
        {"job_id": "j1", "stage_name": "Chunk", "status": "failed", "error": "boom"}
    ]


def test_postgres_driver_url_pins_asyncpg_for_every_postgres_scheme():
    """HIGH-05: ``postgres://`` (the alias Heroku / Render / Supabase hand out) and a sync-driver URL both
    become asyncpg URLs; the password survives."""
    pytest.importorskip("pgvector")
    from tarnrag.storage.repository.postgres import PostgresRepository

    repo = PostgresRepository("postgresql://u:p@h/db", embedding_dimension=3)  # no connection is made
    for url in ("postgres://u:p@h:5432/db", "postgresql://u:p@h:5432/db", "postgresql+psycopg2://u:p@h:5432/db"):
        assert repo._driver_url(url) == "postgresql+asyncpg://u:p@h:5432/db"


async def test_hydrate_reads_the_chunks_table_once(repo):
    """MED-08: text, licensing and provenance come back from ONE ``chunks`` query (the provenance columns
    ride along) — not a second ``chunks`` round trip for the provenance."""
    from sqlalchemy import event

    _, ids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}), [_chunk("a", 0, 2), _chunk("b", 1, 2)]
    )
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(repo.engine.sync_engine, "before_cursor_execute", _record)
    try:
        records = await repo.hydrate(list(ids))
    finally:
        event.remove(repo.engine.sync_engine, "before_cursor_execute", _record)
    assert [r.chunk_id for r in records] == list(ids)
    assert sum(1 for s in statements if "FROM chunks" in s) == 1


async def test_clearing_a_document_is_not_bounded_by_the_bind_parameter_limit(repo):
    """MED-08: clearing a document's search-index rows uses a subquery, so it doesn't expand one bind
    parameter per chunk (which fails past SQLITE_MAX_VARIABLE_NUMBER — lowered here to make it cheap)."""
    import sqlite3

    from sqlalchemy import event

    @event.listens_for(repo.engine.sync_engine, "connect")
    def _low_limit(dbapi_connection, _record):
        dbapi_connection.driver_connection._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 50)

    await repo.engine.dispose()  # new connections pick up the low limit
    chunks = [_chunk(f"chunk {i}", i, 120) for i in range(120)]
    await repo.store_document_with_chunks(Document(content="d", metadata={"source_id": "big"}), chunks)
    assert await repo.delete_document("big") is True
    async with repo.engine.connect() as conn:
        assert (await conn.exec_driver_sql("SELECT count(*) FROM fts_chunks")).fetchone()[0] == 0
