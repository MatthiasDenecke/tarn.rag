"""SQLAlchemy 2.0 Core repository shared by ingestion and retrieval.

All dialect-agnostic table definitions and CRUD live in ``DocumentRepository``;
subclasses supply only the Postgres/SQLite specifics via a small set of hooks.
"""

from __future__ import annotations

import statistics
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from tarnrag.core.engine.config import DatabaseSettings
from tarnrag.core.hashing import compute_content_hash
from tarnrag.storage.repository import chunk_provenance as cp
from tarnrag.contracts import (
    Candidate,
    Chunk,
    ChunkStore,
    CorpusStatus,
    Document,
    DocumentFacts,
    DocumentFactsSource,
    JobStatusSource,
    RetrievalStore,
)
from tarnrag.storage.repository.read_assembly import ReadAssembler
from tarnrag.storage.status import DocumentStatusReader

# §8 license_class is a closed enum (matches the strategy doc). Single source of truth for the
# CHECK constraints on documents.license_class / chunks.license_class.
LICENSE_CLASSES = (
    "customer_licensed",
    "public_domain",
    "modusq_authored",
    "third_party_copyrighted",
    "third_party_licensed",
)
_LICENSE_CHECK = "license_class IN (" + ", ".join(f"'{c}'" for c in LICENSE_CLASSES) + ")"

# Provenance columns carried in a DTO's ``metadata`` bag — the single source of truth for the
# bag<->column mapping, used in BOTH directions: each entry's callable derives the column value
# from the metadata bag on write, and (since column name == metadata key) the same name reads the
# column back into the bag. Adding a provenance field is one entry here, not edits in four methods.
# The identity (``document_id``), ``content`` / ``ordinal``, and the computed chunk ``content_hash``
# are NOT passthrough, so they stay explicit in the write/read methods below.
_DOC_PROVENANCE = {
    "title": lambda md: md.get("title"),
    "source_kind": lambda md: md.get("source_kind") or "document",
    "standard_id": lambda md: md.get("standard_id"),
    "doc_version": lambda md: md.get("doc_version"),
    "license_class": lambda md: md.get("license_class") or "public_domain",
    "content_hash": lambda md: md.get("content_hash"),
}
_CHUNK_PROVENANCE = {
    "locator": lambda md: md.get("locator"),
    "license_class": lambda md: md.get("license_class") or "public_domain",
    "ai_grounding_allowed": lambda md: int(md.get("ai_grounding_allowed", 1)),
    "available": lambda md: int(md.get("available", 1)),
}


# Namespace for the content-addressed chunk ids (``DocumentRepository._chunk_id``) — fixed forever:
# changing it would re-key every chunk a re-ingest writes.
_CHUNK_ID_NAMESPACE = uuid.UUID("3f0d1c2a-7b4e-5d6f-8a9b-0c1d2e3f4a5b")


class DocumentRepository(ChunkStore, RetrievalStore, JobStatusSource, DocumentFactsSource):
    """
    SQLAlchemy Core repository shared by ingestion and retrieval.

    Subclasses supply only the dialect specifics: the async driver URL, the vector
    column type + value encoding, the upsert, dialect-only schema objects (pgvector
    extension/index, expression indexes), and the vector-search query. Everything
    else — table definitions, lifecycle, and the rest of the CRUD — is shared here.

    Guarantees:
    - Atomicity: multi-row writes (store_document_with_chunks / store_chunks /
      store_embeddings) run in one transaction (engine.begin) — all or nothing.
    - Idempotency: documents are keyed by metadata['source_id'] (UNIQUE). Re-ingesting
      a source UPSERTS the document and REPLACES its chunks/embeddings (cascade);
      never duplicates.
    """

    @classmethod
    async def create(
        cls, database: DatabaseSettings, embedding_dimension: int
    ) -> DocumentRepository:
        """Build and connect the repository selected by ``database.document_url`` (Postgres on
        a ``postgres`` URL, else SQLite). Heavy backends are imported lazily."""
        if "postgres" in database.document_url:
            from tarnrag.storage.repository.postgres import PostgresRepository

            repo: DocumentRepository = PostgresRepository(
                database.document_url, embedding_dimension=embedding_dimension
            )
        else:
            from tarnrag.storage.repository.sqlite import SqliteRepository

            repo = SqliteRepository(
                database.document_url, embedding_dimension=embedding_dimension
            )
        await repo.connect()
        return repo

    def __init__(self, connection_url: str, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
        self.engine: AsyncEngine = create_async_engine(self._driver_url(connection_url))
        self.metadata = MetaData()
        self._define_tables()
        # The read-side assembly companion (child-table fetches + ChunkProvenance/ChunkRecord
        # construction) — split out of this class so it stays the port implementation.
        self.reads = ReadAssembler(
            self.chunks, self.table_cells, self.chunk_annotations, self.method_chunks
        )

    # ---------------- dialect hooks (subclasses implement) ----------------

    @abstractmethod
    def _driver_url(self, url: str) -> str:
        """Map a user URL to its async driver URL (e.g. postgresql+asyncpg://...)."""

    @abstractmethod
    def _vector_type(self):
        """Column type for an embedding vector (pgvector Vector vs Text/JSON)."""

    @abstractmethod
    def _encode_vector(self, vector: list[float]):
        """Adapt a vector for storage (list for pgvector, JSON string for SQLite)."""

    @abstractmethod
    def _decode_vector(self, stored) -> list[float]:
        """Inverse of _encode_vector when reading rows back."""

    @abstractmethod
    def _dialect_insert(self, table: Table) -> Any:
        """The dialect's ``INSERT`` construct for ``table`` (``sqlalchemy.dialects.<dialect>.insert``) —
        both SQLite and Postgres support ``on_conflict_do_update``, which ``_upsert`` builds on."""

    # dense_knn / sparse_search / hydrate are the RetrievalStore port (contracts.ports); each dialect
    # implements them. Schema hooks: default no-ops. The pgvector extension must exist *before* the
    # tables (the embeddings vector column needs it); expression/vector indexes must
    # be created *after* the tables exist — hence two hooks rather than one.
    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        return None

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        return None

    # ---------------- shared schema ----------------

    def _define_tables(self) -> None:
        # §8 documents (ModusQ §8.1): typed provenance + ``document_id`` (== source_id) as the PK.
        # ``content`` (full doc text) and ``content_hash`` are Python-side extras beyond the bare
        # §8 columns — harmless to the C++ reader, which reads only the columns it knows.
        self.documents = Table(
            "documents",
            self.metadata,
            Column("document_id", Text, primary_key=True),  # == source_id (the public handle)
            Column("content", Text, nullable=False),         # full doc text (Python-side; not §8)
            Column("title", Text),
            Column("source_kind", Text, nullable=False, default="document"),
            Column("standard_id", Text),
            Column("doc_version", Text),
            Column("license_class", Text, nullable=False, default="public_domain"),
            Column("content_hash", Text, index=True),        # content-dedup key
            CheckConstraint(_LICENSE_CHECK, name="ck_documents_license_class"),
        )
        # §8 chunks: typed provenance + license denormalized for fast filtering. No metadata bag —
        # a positional/metadata field returns later (see the rag-chunk-metadata-deferred note).
        self.chunks = Table(
            "chunks",
            self.metadata,
            Column("chunk_id", Text, primary_key=True),
            Column(
                "document_id",
                Text,
                ForeignKey("documents.document_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("ordinal", Integer, nullable=False),      # position within the document
            Column("text", Text, nullable=False),            # canonical chunk text (returned verbatim)
            Column("locator", Text),                         # citable locator, e.g. '§6.4.2'
            Column("license_class", Text, nullable=False, default="public_domain"),
            Column("ai_grounding_allowed", Integer, nullable=False, default=1),
            Column("available", Integer, nullable=False, default=1),
            Column("content_hash", Text, nullable=False),    # sha256 of the chunk text
            # layout-aware provenance (re-added after the §8 metadata-bag drop):
            Column("header_path", Text),                      # JSON list[str] — the section breadcrumb
            Column("level", Integer, nullable=False, default=0),  # auto-merging tree: 0 = leaf, >0 = section parent
            Column("parent_chunk_id", Text),                  # the section parent's chunk_id (soft self-ref, no FK)
            Column("geometry", Text),                         # JSON Geometry — char spans (+ PDF page boxes)
            CheckConstraint(_LICENSE_CHECK, name="ck_chunks_license_class"),
            CheckConstraint("ai_grounding_allowed IN (0, 1)", name="ck_chunks_ai_grounding"),
            CheckConstraint("available IN (0, 1)", name="ck_chunks_available"),
            Index("idx_chunks_document", "document_id"),
            Index("idx_chunks_license", "license_class", "available"),
            Index("idx_chunks_parent", "parent_chunk_id"),   # merge-up: leaf -> section parent
        )
        # Cell-level table structure for table chunks — enough to cite/highlight a cell and to query
        # by row/column header id (the layout-aware requirement). One row per cell; cascades with the
        # chunk. The table's markdown lives in the chunk's ``text``; n_rows/n_cols derive from the cells.
        self.table_cells = Table(
            "table_cells",
            self.metadata,
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("cell_id", Text, nullable=False),          # the TableCell id within its table
            Column("row", Integer, nullable=False),
            Column("col", Integer, nullable=False),
            Column("row_span", Integer, nullable=False, default=1),
            Column("col_span", Integer, nullable=False, default=1),
            Column("is_column_header", Integer, nullable=False, default=0),
            Column("is_row_header", Integer, nullable=False, default=0),
            Column("text", Text, nullable=False, default=""),
            Column("geometry", Text),                         # JSON Geometry for the cell
            PrimaryKeyConstraint("chunk_id", "cell_id"),
            Index("idx_table_cells_chunk", "chunk_id"),
        )
        # Enricher annotations on a chunk (NER / topic / classification …) — one row per annotation,
        # cascades with the chunk. ``type`` is indexed for filter-by-annotation retrieval; ``span`` is
        # the optional sub-region (JSON Geometry); ``deterministic`` flags generative findings (FR-5.3).
        self.chunk_annotations = Table(
            "chunk_annotations",
            self.metadata,
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("ordinal", Integer, nullable=False),       # position within the chunk's annotations
            Column("producer", Text, nullable=False),         # the enricher's name
            Column("type", Text, nullable=False),             # "entity" | "topic" | "classification" | …
            Column("value", Text, nullable=False, default="{}"),  # JSON payload
            Column("span", Text),                             # JSON Geometry sub-span; null = whole chunk
            Column("deterministic", Integer, nullable=False, default=1),
            PrimaryKeyConstraint("chunk_id", "ordinal"),
            Index("idx_chunk_annotations_chunk", "chunk_id"),
            Index("idx_chunk_annotations_type", "type"),
        )
        # §8 method_chunks: resolved reference bundles (method version -> chunk).
        self.method_chunks = Table(
            "method_chunks",
            self.metadata,
            Column("method_id", Text, nullable=False),
            Column("method_version", Text, nullable=False),
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                nullable=False,
            ),
            PrimaryKeyConstraint("method_id", "method_version", "chunk_id"),
            Index("idx_method_chunks_chunk", "chunk_id"),
        )
        # Schema v2: embeddings are strictly 1:1 with chunks (``chunk_id`` IS the identity — the same
        # shape as SQLite's ``vec_chunks``). Embedder identity (model/dimension/…) is index-wide and
        # lives in ``index_meta``, never per row; the historical surrogate ``id`` + ``model`` /
        # ``dimension`` columns are gone (multi-model corpora would be a designed vector-spaces
        # feature, not schema slack).
        self.embeddings = Table(
            "embeddings",
            self.metadata,
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                primary_key=True,
            ),
            Column("vector", self._vector_type(), nullable=False),
        )
        # Document-keyed status PROJECTION for the API (the queue itself is pgQueuer's).
        self.job_status = Table(
            "job_status",
            self.metadata,
            Column("job_id", Text, primary_key=True),
            Column("document_id", Text, nullable=False, index=True),
            Column("stage_name", Text, nullable=False),
            Column("status", Text, nullable=False, default="queued"),
            Column("error", Text),
            Column("created_at", TIMESTAMP, server_default=func.now()),
            Column("updated_at", TIMESTAMP, server_default=func.now()),
        )
        # §8 build/compatibility metadata (key/value): schema/ingestion versions and the
        # embedding-pipeline fingerprint the retrieval store validates at open(). Dialect-agnostic —
        # the first §8 table to live in the base as the repository takes over the retrieval index.
        self.index_meta_table = Table(
            "index_meta",
            self.metadata,
            Column("key", Text, primary_key=True),
            Column("value", Text, nullable=False),
        )

    # ---------------- lifecycle ----------------

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await self._before_create_schema(conn)
            await conn.run_sync(self.metadata.create_all)
            await self._after_create_schema(conn)

    async def disconnect(self) -> None:
        await self.engine.dispose()

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ---------------- §8 index metadata ----------------

    async def write_index_meta(self, meta: dict[str, str]) -> None:
        """
        Upsert §8 build/compatibility metadata (key/value) — e.g. the embedding-pipeline
        fingerprint the retrieval store validates at ``open()``. Atomic upsert (``_upsert``); takes a
        plain dict so the producer owns how the fingerprint is assembled.
        """
        async with self.engine.begin() as conn:
            for key, value in meta.items():
                await self._upsert(
                    conn, self.index_meta_table, self.index_meta_table.c.key, key, {"value": value}
                )

    async def index_meta(self) -> dict[str, str]:
        """
        All §8 build/compatibility metadata as a dict (empty before anything is written).
        """
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(self.index_meta_table))).all()
        return {key: value for key, value in rows}

    # ---------------- writes (shared Core) ----------------

    async def store_document(self, doc: Document) -> str:
        """
        Atomic: upsert + chunk-delete share one engine.begin() transaction — if the delete
        fails, the upsert is rolled back too (commits only on clean exit).
        """
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-storing REPLACES derived data (idempotency): clear the search indexes (vec0/FTS
            # don't cascade) then drop chunks (ON DELETE CASCADE removes embeddings).
            await self._clear_chunk_index(conn, doc_id)
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.document_id == doc_id)
            )
            return doc_id

    async def store_documents(self, docs: list[Document]) -> list[str]:
        """Bulk ``store_document``: upsert every document — each replacing its derived chunks/search indexes
        — in **one** transaction, so a batch of documents costs a single commit instead of one per document
        (the ingest hot path). Reuses the exact per-document logic; result is identical to looping
        ``store_document``."""
        async with self.engine.begin() as conn:
            ids: list[str] = []
            for doc in docs:
                doc_id = await self._upsert_document(conn, self._doc_values(doc))
                await self._clear_chunk_index(conn, doc_id)
                await conn.execute(self.chunks.delete().where(self.chunks.c.document_id == doc_id))
                ids.append(doc_id)
            return ids

    async def store_document_with_chunks(
        self, doc: Document, chunks: list[Chunk]
    ) -> tuple[str, list[str]]:
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-ingest replaces chunks; clear non-cascading search indexes first.
            await self._clear_chunk_index(conn, doc_id)
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.document_id == doc_id)
            )
            chunk_ids = await self._insert_chunks(conn, doc_id, chunks)
            return doc_id, chunk_ids

    async def store_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Persist a batch of chunks atomically, **replacing** whatever chunks their documents already
        hold — idempotent, so a retried Chunk job (a worker that died after this commit but before its
        job was acked, or a requeue after a downstream failure) rewrites the same chunk set instead of
        adding a second one. A document's chunks always arrive together (one Chunk job per document);
        chunk ids are content-addressed (see ``_chunk_id``), so the retry reproduces the very ids its
        first attempt handed to the Embed jobs."""
        async with self.engine.begin() as conn:
            for document_id in dict.fromkeys(ch.parent_doc_id for ch in chunks):
                await self._clear_chunk_index(conn, document_id)
                await conn.execute(self.chunks.delete().where(self.chunks.c.document_id == document_id))
            return await self._insert_chunks(conn, None, chunks)

    # ``store_embeddings`` is dialect-owned (the ``ChunkStore`` port keeps it abstract here):
    # SQLite writes the sqlite-vec ``vec_chunks`` virtual table; Postgres upserts the
    # ``embeddings`` table (``ON CONFLICT (chunk_id)``). Both are idempotent per chunk.

    async def register_method_bundle(
        self, method_id: str, method_version: str, chunk_ids: list[str]
    ) -> None:
        """Register the §8 method bundle — the resolved set of chunks for ``(method_id,
        method_version)``. REPLACES any existing bundle for that pair in one transaction (idempotent
        re-registration); an empty ``chunk_ids`` just clears it. A retrieval scoped to a matching
        ``MethodRef`` then returns only these chunks (the ``method_scope`` pre-filter). The chunks must
        already be stored (FK). This is the writer the method-scope query path was missing."""
        mc = self.method_chunks.c
        async with self.engine.begin() as conn:
            await conn.execute(
                self.method_chunks.delete().where(
                    (mc.method_id == method_id) & (mc.method_version == method_version)
                )
            )
            if chunk_ids:
                await conn.execute(
                    insert(self.method_chunks),
                    [
                        {"method_id": method_id, "method_version": method_version, "chunk_id": cid}
                        for cid in chunk_ids
                    ],
                )

    async def method_bundle(self, method_id: str, method_version: str) -> list[str]:
        """The chunk ids registered for ``(method_id, method_version)`` (empty if unregistered) —
        the read-back for :meth:`register_method_bundle`, sorted for determinism."""
        mc = self.method_chunks.c
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(mc.chunk_id)
                    .where((mc.method_id == method_id) & (mc.method_version == method_version))
                    .order_by(mc.chunk_id)
                )
            ).all()
        return [r[0] for r in rows]

    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        # No-op BY DESIGN (not unfinished): §8 chunks carry no free-form metadata column. Enrichment
        # now annotates the document (the Enrich stage) and rides into chunks via ChunkProvenance;
        # a general chunk-metadata bag is deferred. The method stays on the port as a latent capability.
        return None

    # ---------------- dialect search-index hooks (vec0 / FTS live outside the FK graph) ----------------

    async def _index_chunk_text(
        self, conn: AsyncConnection, ids: list[str], chunks: list[Chunk]
    ) -> None:
        """
        Index chunk text for sparse search. No-op by default; SQLite writes ``fts_chunks``.
        """
        return None

    async def _clear_chunk_index(self, conn: AsyncConnection, document_id: str) -> None:
        """
        Drop a document's chunks from search indexes the FK CASCADE can't reach (vec0/FTS
        virtual tables). No-op by default (the embeddings table cascades).
        """
        return None

    async def _count_doc_embeddings(self, conn: AsyncConnection, document_id: str) -> int:
        """
        Vector count for a document — default: the embeddings table; SQLite counts ``vec_chunks``.
        """
        return (
            await conn.execute(
                select(func.count())
                .select_from(self.embeddings)
                .join(self.chunks, self.embeddings.c.chunk_id == self.chunks.c.chunk_id)
                .where(self.chunks.c.document_id == document_id)
            )
        ).scalar_one()

    async def _embedding_counts_by_document(self, conn: AsyncConnection) -> dict[str, int]:
        """
        Per-document vector counts — default: the embeddings table; SQLite counts ``vec_chunks``.
        """
        return dict(
            (
                await conn.execute(
                    select(self.chunks.c.document_id, func.count(self.embeddings.c.chunk_id))
                    .select_from(
                        self.chunks.join(
                            self.embeddings,
                            self.embeddings.c.chunk_id == self.chunks.c.chunk_id,
                        )
                    )
                    .group_by(self.chunks.c.document_id)
                )
            ).all()
        )

    # ---------------- reads (shared Core) ----------------

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.documents).where(self.documents.c.document_id == doc_id)
                )
            ).mappings().first()
        if row is None:
            return None
        return Document(id=row["document_id"], content=row["content"], metadata=self._doc_metadata(row))

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.chunks).where(self.chunks.c.chunk_id == chunk_id)
                )
            ).mappings().first()
            if row is None:
                return None
            chunk = self._row_to_chunk(row)
            await self.reads.attach_tables(conn, [chunk])
            await self.reads.attach_annotations(conn, [chunk])
            return chunk

    async def get_chunks_by_document(self, doc_id: str) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(self.chunks)
                    .where(self.chunks.c.document_id == doc_id)
                    .order_by(self.chunks.c.ordinal)
                )
            ).mappings().all()
            chunks = [self._row_to_chunk(r) for r in rows]
            await self.reads.attach_tables(conn, chunks)
            await self.reads.attach_annotations(conn, chunks)
            return chunks

    async def query_chunks(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[Chunk]:
        stmt = select(self.chunks)
        if "source_id" in filters:  # source_id is the document_id under §8
            stmt = stmt.where(self.chunks.c.document_id == filters["source_id"])
        if "license_class" in filters:
            stmt = stmt.where(self.chunks.c.license_class == filters["license_class"])
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.limit(limit))).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    # ---------------- job-status projection (for the document-status API) ----------------

    async def record_job(
        self,
        document_id: str,
        job_id: str,
        stage_name: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Upsert a job's status row (atomic: ``_upsert``)."""
        async with self.engine.begin() as conn:
            await self._upsert(
                conn, self.job_status, self.job_status.c.job_id, job_id,
                {"status": status, "error": error, "updated_at": func.now()},
                insert_extra={"document_id": document_id, "stage_name": stage_name},
            )

    async def record_jobs(
        self, jobs: list[tuple[str, str, str]], status: str, error: str | None = None
    ) -> None:
        """Bulk ``record_job``: upsert many jobs' status rows in **one** transaction — one commit per batch of
        jobs instead of one per job (the ingest hot path). ``jobs`` is ``(document_id, job_id, stage_name)``."""
        async with self.engine.begin() as conn:
            for document_id, job_id, stage_name in jobs:
                await self._upsert(
                    conn, self.job_status, self.job_status.c.job_id, job_id,
                    {"status": status, "error": error, "updated_at": func.now()},
                    insert_extra={"document_id": document_id, "stage_name": stage_name},
                )

    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """Debug-only per-job breakdown for one document."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(
                        self.job_status.c.job_id,
                        self.job_status.c.stage_name,
                        self.job_status.c.status,
                        self.job_status.c.error,
                    ).where(self.job_status.c.document_id == document_id)
                )
            ).mappings().all()
        return [dict(r) for r in rows]

    async def delete_document_jobs(self, document_id: str) -> bool:
        """Remove a document's job_status rows (used when deleting a document)."""
        async with self.engine.begin() as conn:
            return await self._delete_job_rows(conn, document_id)

    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts (presence + chunk/embedding counts) for this document."""
        async with self.engine.connect() as conn:
            present = (
                await conn.execute(
                    select(self.documents.c.document_id)
                    .where(self.documents.c.document_id == document_id)
                )
            ).first() is not None
            if not present:
                return DocumentFacts(present=False, chunk_count=0, embedding_count=0)
            chunk_count, leaf_count = (
                await conn.execute(
                    select(func.count(), func.count().filter(self.chunks.c.level == 0))
                    .select_from(self.chunks)
                    .where(self.chunks.c.document_id == document_id)
                )
            ).one()
            embedding_count = await self._count_doc_embeddings(conn, document_id)
        return DocumentFacts(True, chunk_count, embedding_count, leaf_count)

    async def corpus_stats(self) -> CorpusStatus:
        """A whole-repository snapshot: document / chunk / embedding counts plus the distribution of
        document length in characters (min / median / mean / max / total). The read model behind the
        console's ``status``."""
        docs = await self.list_documents()  # per-document chunk + embedding counts (no N+1)
        lengths = sorted(await self._document_lengths())
        n = len(lengths)
        chunk_count = sum(d["chunk_count"] for d in docs)
        return CorpusStatus(
            document_count=len(docs),
            chunk_count=chunk_count,
            embedding_count=sum(d["embedding_count"] for d in docs),
            total_chars=sum(lengths),
            min_chars=lengths[0] if n else 0,
            max_chars=lengths[-1] if n else 0,
            mean_chars=sum(lengths) / n if n else 0.0,
            median_chars=statistics.median(lengths) if n else 0.0,
            mean_chunks_per_doc=chunk_count / len(docs) if docs else 0.0,
        )

    async def _document_lengths(self) -> list[int]:
        """The character length of each document's stored content (``func.length`` — dialect-agnostic)."""
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(func.length(self.documents.c.content)))).all()
        return [row[0] or 0 for row in rows]

    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids whose stored content_hash matches — content dedup."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(self.documents.c.document_id)
                    .where(self.documents.c.content_hash == content_hash)
                )
            ).all()
        return [r[0] for r in rows]

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete the document; its chunks (and their embeddings) cascade off the FKs, and the
        non-cascading search indexes (vec0/FTS) are cleared first. Returns True if it existed.
        Leaves the job_status rows (use ``delete_document_jobs``, or ``delete_document_and_jobs``
        to remove both atomically).
        """
        async with self.engine.begin() as conn:
            return await self._delete_document_rows(conn, document_id)

    async def delete_document_and_jobs(self, document_id: str) -> bool:
        """
        Delete a document's data AND its job_status rows in **one transaction** (atomic): chunks +
        embeddings cascade off the FKs, the non-cascading vec0/FTS indexes are cleared, and the
        job_status rows are removed — all-or-nothing, so any failure rolls the whole delete back.
        Returns True if anything existed. (Data and job_status share this one repository, so the
        delete that used to span two stores is now a single transaction.)
        """
        async with self.engine.begin() as conn:
            removed_data = await self._delete_document_rows(conn, document_id)
            removed_jobs = await self._delete_job_rows(conn, document_id)
            return removed_data or removed_jobs

    async def _delete_document_rows(self, conn: AsyncConnection, document_id: str) -> bool:
        """
        Delete the document row (chunks/embeddings cascade) and clear the non-cascading vec0/FTS
        indexes, on the given connection. Returns True if the document existed.
        """
        await self._clear_chunk_index(conn, document_id)
        res = await conn.execute(
            self.documents.delete().where(self.documents.c.document_id == document_id)
        )
        return res.rowcount > 0

    async def _delete_job_rows(self, conn: AsyncConnection, document_id: str) -> bool:
        """
        Delete the document's job_status rows on the given connection.
        """
        res = await conn.execute(
            self.job_status.delete().where(self.job_status.c.document_id == document_id)
        )
        return res.rowcount > 0

    async def list_documents(self) -> list[dict[str, Any]]:
        """Inventory of all documents with chunk/embedding counts (three grouped queries — no
        per-document N+1)."""
        async with self.engine.connect() as conn:
            docs = (
                await conn.execute(
                    select(self.documents.c.document_id, self.documents.c.content_hash)
                )
            ).all()
            chunk_counts = dict(
                (
                    await conn.execute(
                        select(self.chunks.c.document_id, func.count())
                        .group_by(self.chunks.c.document_id)
                    )
                ).all()
            )
            embedding_counts = await self._embedding_counts_by_document(conn)
        return [
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "chunk_count": chunk_counts.get(document_id, 0),
                "embedding_count": embedding_counts.get(document_id, 0),
            }
            for document_id, content_hash in docs
        ]

    async def document_status(self, document_id: str) -> dict[str, Any] | None:
        """Convenience: status over this repo alone (job_status + data both here). The
        retrieval path composes a reader over the repo (jobs) + the index store (facts)."""
        return await DocumentStatusReader(self, self).document_status(document_id)

    # ---------------- shared helpers ----------------

    async def _upsert(
        self,
        conn: AsyncConnection,
        table: Table,
        key_column: Any,
        key_value: Any,
        values: dict[str, Any],
        insert_extra: dict[str, Any] | None = None,
    ) -> None:
        """Atomic upsert: ``INSERT`` the row (the key + ``values`` + any insert-only ``insert_extra``)
        ``ON CONFLICT (key) DO UPDATE`` with ``values``. One statement, so two concurrent writers of
        the same key (two workers stamping a fresh index's ``index_meta``, two ingests of one
        ``source_id``) can't both miss an UPDATE and then collide on the INSERT — the race the old
        update-else-insert pair had under READ COMMITTED. The single home for the upsert shared by
        ``write_index_meta`` / ``record_job`` / ``_upsert_document``."""
        stmt = self._dialect_insert(table).values(
            **{key_column.name: key_value, **values, **(insert_extra or {})}
        )
        await conn.execute(stmt.on_conflict_do_update(index_elements=[key_column], set_=values))

    # §8 filtered retrieval over-fetch -------------------------------------
    _OVERFETCH_FACTOR = 4  # ModusQ §5.4: grow the candidate window ×4 until enough permitted hits

    async def _overfetch(
        self,
        k: int,
        total: int,
        page: Callable[[int], Awaitable[list[tuple[str, float]]]],
    ) -> list[Candidate]:
        """Return the top ``k`` permitted candidates for a filtered search, over-fetching to backfill past
        disallowed chunks (ModusQ §5.4). ``page(window)`` runs the dialect's KNN/BM25 over the top-``window``
        raw hits with the permitted predicate applied, returning the surviving ``(chunk_id, raw_score)`` rows
        best-first. The window grows ×4 until it yields ≥ ``k`` or covers the whole index (``total``); the
        result is re-ranked 1..k. (sqlite-vec KNN picks its k nearest *before* a join can filter, so a
        post-filter would under-return; over-fetching backfills.) The single home for the loop both dialects
        share."""
        if k <= 0:
            return []
        window = k * self._OVERFETCH_FACTOR
        while True:
            rows = await page(window)
            if len(rows) >= k or window >= total:
                return [
                    Candidate(chunk_id=cid, rank=i + 1, raw_score=score)
                    for i, (cid, score) in enumerate(rows[:k])
                ]
            window *= self._OVERFETCH_FACTOR

    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        """Upsert on the ``document_id`` primary key. Returns the document_id so chunks resolve
        their FK."""
        document_id = values["document_id"]
        await self._upsert(
            conn, self.documents, self.documents.c.document_id, document_id,
            {k: v for k, v in values.items() if k != "document_id"},
        )
        return document_id

    def _doc_values(self, doc: Document) -> dict:
        """Map a Document DTO to §8 document columns: identity + ``content`` explicit, the rest of
        the provenance driven by ``_DOC_PROVENANCE`` (the bag<->column source of truth)."""
        md = doc.metadata or {}
        return {
            "document_id": md.get("source_id") or doc.id or str(uuid.uuid4()),
            "content": doc.content,
            **{col: derive(md) for col, derive in _DOC_PROVENANCE.items()},
        }

    async def _insert_chunks(
        self, conn: AsyncConnection, document_id: str | None, chunks: list[Chunk]
    ) -> list[str]:
        # Pass 1: assign ids and map (document, ordinal) -> chunk_id, so a child resolves its parent.
        ids = [ch.id or self._chunk_id(document_id or ch.parent_doc_id, ch) for ch in chunks]
        id_by_ordinal = {
            (document_id or ch.parent_doc_id, ch.chunk_index): cid for ch, cid in zip(chunks, ids)
        }
        # Pass 2: build the chunk rows (resolving parent_chunk_id) + the table_cell rows.
        rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        annotation_rows: list[dict[str, Any]] = []
        for ch, cid in zip(chunks, ids):
            doc = document_id or ch.parent_doc_id
            md = ch.metadata or {}
            prov = ch.provenance
            parent_ordinal = md.get("parent_ordinal")
            parent_chunk_id = id_by_ordinal.get((doc, parent_ordinal)) if parent_ordinal is not None else None
            rows.append(
                {
                    "chunk_id": cid,
                    "document_id": doc,
                    "ordinal": ch.chunk_index,
                    "text": ch.content,
                    **{col: derive(md) for col, derive in _CHUNK_PROVENANCE.items()},
                    "content_hash": compute_content_hash(ch.content),
                    **cp.provenance_columns(prov, parent_chunk_id),
                }
            )
            if prov and prov.table:
                cell_rows.extend(cp.table_cell_rows(cid, prov.table))
            if prov and prov.annotations:
                annotation_rows.extend(cp.annotation_rows(cid, prov.annotations))
        if rows:
            await conn.execute(insert(self.chunks), rows)
            if cell_rows:
                await conn.execute(insert(self.table_cells), cell_rows)
            if annotation_rows:
                await conn.execute(insert(self.chunk_annotations), annotation_rows)
            await self._index_chunk_text(conn, ids, chunks)
        return ids

    @staticmethod
    def _chunk_id(document_id: str, chunk: Chunk) -> str:
        """A content-addressed chunk id: ``uuid5`` of (document, ordinal, content hash). Deterministic, so
        re-running the same Chunk job reproduces the same ids (idempotent retries — in-flight Embed jobs
        from the first attempt still point at real chunks); content-bearing, so a re-ingest that changes a
        chunk's text gives it a new id and a citation to the old text can't silently resolve to new text."""
        key = f"{document_id}\x1f{chunk.chunk_index}\x1f{compute_content_hash(chunk.content)}"
        return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key))

    def _row_to_chunk(self, r) -> Chunk:
        # total_chunks is left at its None default: §8 stores ordinal, not the total.
        return Chunk(
            id=r["chunk_id"],
            parent_doc_id=r["document_id"],
            content=r["text"],
            chunk_index=r["ordinal"],
            provenance=cp.row_to_provenance(r),
            metadata=self._chunk_metadata(r),
        )

    @staticmethod
    def _doc_metadata(r) -> dict[str, Any]:
        """Reconstruct a metadata dict from §8 document columns (inverse of ``_doc_values``)."""
        return {"source_id": r["document_id"], **{col: r[col] for col in _DOC_PROVENANCE}}

    @staticmethod
    def _chunk_metadata(r) -> dict[str, Any]:
        """Reconstruct a metadata dict from §8 chunk columns: identity + the ``_CHUNK_PROVENANCE``
        passthrough + the computed ``content_hash`` column."""
        return {
            "source_id": r["document_id"],
            **{col: r[col] for col in _CHUNK_PROVENANCE},
            "content_hash": r["content_hash"],
        }
