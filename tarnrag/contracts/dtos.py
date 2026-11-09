"""Cross-boundary data contracts (DTOs).

At the core are four models that are TWO FORMS of the things being ingested — not four
unrelated types, and not an inheritance hierarchy:

* IN-FLIGHT form -> ``PipelineItem``. The single, uniform type that flows between
  stages. A *document* (at Load/Clean) and a *chunk* (at Chunk/Enrich/Embed) are
  BOTH just ``PipelineItem``s; which one it is rides in ``metadata`` (e.g.
  ``chunk_index``), not in the type. The engine (stages, worker, DAG, inline job
  payload) is generic over this one type, so it stays stage- and entity-agnostic.

* AT-REST form -> ``Document`` / ``Chunk`` / ``Embedding``. Typed persistence records
  (DTOs) materialised at the ResultSinks to write rows (and reconstructed on reads).
  They NEVER flow; the persistent schema is the SQLAlchemy ``Table``s in
  ``DocumentRepository``, which maps each DTO to/from its table.

Worked trace of one ingest — watch the form change:

    after Load    PipelineItem(content="<doc text>", metadata={source_id, doc_id})
    after Chunk   PipelineItem(content="<chunk 3>",  metadata={..., chunk_index: 3})
                      |  ChunkResultSink persists it ->
                      v
                  Chunk(id="c3", parent_doc_id="d1", content="<chunk 3>", chunk_index=3)
    after Embed   Embedding(chunk_id="c3", vector=[...])   # terminal: never flows

So a chunk-in-flight IS a ``PipelineItem``; ``Chunk`` is its stored row. ``Embedding``
is the exception — Embed is the terminal stage, so an embedding is only ever at-rest.

Why two forms, not one: the in-flight form is deliberately loose (one uniform type +
a metadata bag) so the engine is generic and extensible; the at-rest form is typed so
the storage contract is explicit and validated (``store_chunks`` guarantees real
``parent_doc_id`` / ``chunk_index`` fields, not "hopefully in a dict"). The field
overlap (id/content/metadata) is what makes them look alike, but it is coincidental
and diverges (``Chunk`` adds FKs, ``Embedding`` has no content) — hence no shared base
or inheritance. The relations that DO hold are composition (``Chunk.parent_doc_id`` ->
``Document``, ``Embedding.chunk_id`` -> ``Chunk``) and transformation (a sink maps a
``PipelineItem`` into a DTO) — expressed by fields and functions, not a hierarchy.

Two small value records round out the module: ``DocumentFacts`` (a document's presence +
chunk/embedding counts, the facts port's return type) and ``MethodRef`` (a reference to an
indexed method, used in both queries and results).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tarnrag.contracts.structure import ChunkProvenance, StructuredDocument


class PipelineItem(BaseModel):
    """
    The IN-FLIGHT form — the single, uniform type that flows through every stage.

    A document or a chunk *in flight* is a PipelineItem (which one rides in
    ``metadata``). The engine (stages, worker, DAG, inline job payload) is generic
    over this one type. Distinct from — and not a base of — the at-rest DTOs below,
    which never flow (see the module docstring for the in-flight/at-rest framing).
    """

    id: str | None = None  # assigned by the storage layer
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Document-phase payload (set by the structured-extraction stage, refined by enrichers); the
    # chunk-phase payload (set by the Chunk stage) is ``provenance``. Both default None, so the
    # plain-text path and every existing caller are unaffected. See ``contracts/structure.py``.
    document: StructuredDocument | None = None
    provenance: ChunkProvenance | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def derive(
        self,
        *,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        document: StructuredDocument | None = None,
    ) -> PipelineItem:
        """A fresh item carrying forward ``document`` + ``provenance``; ``content`` / ``metadata`` (and,
        for a stage that rewrites the document itself, ``document``) change — each defaults to this
        item's. ``id`` is reset (the worker re-ids the output), so a 1→1 stage emits a new item without
        re-listing the flow-through fields — the one place they're threaded, so adding a flow-through
        field updates this method, not every stage."""
        return PipelineItem(
            content=self.content if content is None else content,
            metadata=self.metadata if metadata is None else metadata,
            document=self.document if document is None else document,
            provenance=self.provenance,
        )


class Document(BaseModel):
    """
    A source document — AT-REST form (persistence DTO; built at a sink, never
    flows). Mapped to/from the ``documents`` table by the repository.
    """

    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Chunk(BaseModel):
    """
    A text chunk — at-rest form (persistence DTO) of a chunk that flowed as a
    PipelineItem. Mapped to/from the ``chunks`` table.
    """

    id: str | None = None
    parent_doc_id: str
    content: str
    chunk_index: int
    # In-flight bookkeeping (set by ChunkStage). §8 stores only the per-chunk ``ordinal``, not
    # the total, so this is None when a Chunk is reconstructed from a stored row.
    total_chunks: int | None = None
    # The layout-aware provenance (geometry / header_path / level / parent / table) the chunker
    # produced; persisted to the chunks columns + table_cells, and reconstructed on read.
    provenance: ChunkProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Embedding(BaseModel):
    """
    A chunk's dense vector — at-rest ONLY (terminal stage output; never flows). A write-side DTO:
    nothing reconstructs an ``Embedding`` from storage.

    The embedding vectors themselves are stored in both backends, in the SAME shape — one store is
    one vector space (the ``index_meta`` fingerprint enforces one embedder per index), so a stored
    embedding is exactly its chunk key + its vector, keyed 1:1. It is the terminal output of the
    ingestion pipeline:

    EmbedStage ──yields──▶  Embedding(chunk_id, vector)     ← the DTO
                                  │
                                  ▼  EmbeddingResultSink → repo.store_embeddings([...])
                  ┌───────────────┴────────────────┐
              Postgres                          SQLite
    UPSERT INTO embeddings            INSERT OR REPLACE INTO vec_chunks
    (chunk_id PK, vector ←pgvector)   (chunk_id PK, embedding — the sqlite-vec
                                       virtual table)

    Embedder identity (model / revision / dimension / …) is NOT per-row data: it lives once per
    index in ``index_meta``, where the fingerprint gate validates it at ``open()``. (Per-row
    ``id`` / ``model`` / ``dimension`` columns existed historically on Postgres only — schema v2
    removed them; multiple models per corpus would be a designed "vector spaces" feature, not a
    schema leniency.)

    Why the DTO is "write-only": retrieval never reads Embedding objects back. Dense search (dense_knn) queries the vector
    index directly — sqlite-vec's MATCH / pgvector's cosine operator — and gets back (chunk_id, distance) pairs; the text
    and provenance are then hydrated from the chunks table. No code path ever materializes a stored vector back into an
    Embedding.
    """

    chunk_id: str
    vector: list[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True)
class DocumentFacts:
    """
    Persisted-data facts for one document: whether its row exists, and how many
    chunks/embeddings it has. ``leaf_count`` is how many of the chunks are retrieval units (leaves —
    the ones that get a vector; section parents don't): the document is fully indexed once
    ``embedding_count`` reaches it. ``None`` (a store that doesn't distinguish) ⇒ every chunk counts.
    """

    present: bool  # the document row exists in the data store
    chunk_count: int  # every stored chunk, section parents included
    embedding_count: int
    leaf_count: int | None = None  # the retrieval units among them (None ⇒ chunk_count)


@dataclass(frozen=True)
class CorpusStatus:
    """A snapshot of the whole document repository: document / chunk / embedding counts and the
    distribution of document length (in characters). The corpus-level companion to ``DocumentFacts``
    (which is per-document) — the read model behind a ``status`` summary."""

    document_count: int
    chunk_count: int
    embedding_count: int
    total_chars: int
    min_chars: int
    max_chars: int
    mean_chars: float
    median_chars: float
    mean_chunks_per_doc: float


@dataclass(frozen=True)
class MethodRef:
    """
    Reference to an indexed method — id plus optional version (``None`` → latest in the index).
    """

    method_id: str
    method_version: str | None = None  # None → latest in the index
