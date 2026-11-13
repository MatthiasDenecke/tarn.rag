"""Cross-boundary ports (interfaces), all implemented by ``DocumentRepository``.

The seams the rest of the system depends on: ``ChunkStore`` (where the ingestion ResultSinks
write document/chunk/embedding data) and the two status ports the ``DocumentStatusReader``
composes — ``DocumentFactsSource`` (presence + counts + content-hash + the list/delete admin ops)
and ``JobStatusSource`` (the per-job rows). Kept as narrow ABCs so each caller depends only on
the slice it uses (ISP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tarnrag.contracts.dtos import Chunk, Document, DocumentFacts, Embedding
from tarnrag.contracts.results import Candidate, ChunkFilter, ChunkRecord


class ChunkStore(ABC):
    """
    The persistence port a ResultSink writes through — implemented by ``DocumentRepository``.
    Methods are async to match the sinks' ``await``ed ``_persist``.
    """

    @abstractmethod
    async def store_document(self, doc: Document) -> str:
        """Upsert a document; return its id (threaded forward as ``metadata['doc_id']``)."""

    async def store_documents(self, docs: list[Document]) -> list[str]:
        """Bulk ``store_document``; return ids in order. Default loops (one txn each); ``DocumentRepository``
        overrides it with a single-transaction bulk upsert (the ingest hot path)."""
        return [await self.store_document(doc) for doc in docs]

    @abstractmethod
    async def refresh_documents(self, docs: list[Document]) -> list[str]:
        """Upsert the document rows **only** — their chunks/embeddings are left alone (unlike
        ``store_documents``, which replaces them). For a stage that rewrites an already stored document's
        text (CleanAndNormalize): a replayed job then can't wipe the chunks and vectors built after it.
        Return ids in order."""

    @abstractmethod
    async def store_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Persist chunks atomically, replacing any chunks their documents already hold (idempotent
        on retry); return their ids (in order)."""

    @abstractmethod
    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        """Persist chunk embeddings; return the chunk ids written."""

    @abstractmethod
    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        """Merge enrichment into a chunk (no-op for stores without a chunk-metadata column)."""


class DocumentFactsSource(ABC):
    """
    Port over the document data store (the repository): status facts, content-hash lookup, and
    the list/delete admin ops.
    """

    @abstractmethod
    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts for a document (from the repository)."""

    @abstractmethod
    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids whose stored ``content_hash`` matches — for content dedup
        (independent of the id policy). Empty if none; possibly several when identical content
        was ingested under different ids."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Remove a document and its derived data (chunks/embeddings/index rows). Returns True
        if a document existed, False if it was unknown."""

    @abstractmethod
    async def list_documents(self) -> list[dict[str, Any]]:
        """Inventory of stored documents — one dict per document with keys ``document_id``,
        ``content_hash``, ``chunk_count``, ``embedding_count``. Order is unspecified."""


class JobStatusSource(ABC):
    """
    Port over the job_status projection (lives on the repository) — read and delete a document's
    per-job rows.
    """

    @abstractmethod
    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """The per-job rows for a document (the job_status projection)."""

    @abstractmethod
    async def delete_document_jobs(self, document_id: str) -> bool:
        """Remove a document's job_status rows (when deleting a document). Returns True if any
        rows were removed."""


class RetrievalStore(ABC):
    """
    The narrow READ port retrieval depends on (ISP) — dense KNN, sparse (BM25) search, and hydration.
    Implemented by ``DocumentRepository`` (sqlite-vec / FTS5 on SQLite, pgvector / tsvector on Postgres);
    the write counterpart is ``ChunkStore``. Retrievers receive it via the ``RetrievalContext``.
    """

    @abstractmethod
    async def dense_knn(
        self, query_vec: list[float], k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """§8 dense retrieval: the nearest ``k`` chunks to ``query_vec`` as ranked ``Candidate``s
        (sqlite-vec on SQLite, pgvector on Postgres). With a ``filter`` the result holds the ``k`` nearest
        *permitted* chunks — disallowed ones are dropped inside the search and over-fetch backfills past
        them (ModusQ §5.4/§5.6), so a tight scope still yields up to ``k`` hits. ``None`` ⇒ no filter."""

    @abstractmethod
    async def sparse_search(
        self, query_text: str, k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """§8 sparse retrieval: the top ``k`` chunks for ``query_text`` by lexical relevance, ranked
        best-first (FTS5 BM25 on SQLite, tsvector / ts_rank on Postgres). ``raw_score`` is the
        dialect-native relevance. ``filter`` is applied as in :meth:`dense_knn` (pre-filter + over-fetch)."""

    @abstractmethod
    async def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """§8 hydration: canonical text + provenance + license for the given chunk ids, preserving
        input order."""
