"""Result sinks — the output side of a stage (D4).

A worker ``submit``s produced results and ``close``s (both synchronous, buffer-only);
the orchestrator later ``await``s ``finalize`` to persist. One sink per stage;
``create_sink_registry`` maps stage name -> sink class.

ID threading: storage ids are threaded forward via metadata (stages propagate it):
``DocumentResultSink`` writes ``metadata['doc_id']``, ``ChunkResultSink`` writes
``metadata['chunk_id']`` — so the FK chain (chunk→doc, embedding→chunk) resolves
without relying on ``item.id`` or the repository honoring caller-supplied ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tarnrag.contracts import Chunk, ChunkStore, Document


@dataclass
class FinalizationOutcome:
    """
    Result of a sink ``finalize`` — whether the buffer persisted, plus an error detail on failure.
    """

    persisted: bool
    detail: str | None = None


class ResultSink(ABC):
    """
    submit()/close() are sync (buffer only); finalize() is async (persists).
    """

    @abstractmethod
    def submit(self, results: list[Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> FinalizationOutcome: ...


class _BufferingSink(ResultSink):
    """
    Shared buffering: submit() appends, close() marks done, finalize() persists the
    whole buffer via _persist() and reports the outcome.
    """

    def __init__(self, repository: ChunkStore):
        self.repo = repository
        self._buffer: list[Any] = []
        self._closed = False

    def submit(self, results: list[Any]) -> None:
        self._buffer.extend(results)

    def close(self) -> None:
        self._closed = True

    async def finalize(self) -> FinalizationOutcome:
        try:
            await self._persist(self._buffer)
            return FinalizationOutcome(persisted=True)
        except Exception as e:  # noqa: BLE001 - report any persistence failure to the orchestrator
            return FinalizationOutcome(persisted=False, detail=str(e))

    @abstractmethod
    async def _persist(self, results: list[Any]) -> None: ...


class PassthroughSink(_BufferingSink):
    """
    For pure-transform stages with no storage target (e.g. Enrich, which annotates the in-flight document).
    Persists nothing; produced items still flow downstream via the orchestrator.
    """

    async def _persist(self, results: list[Any]) -> None:
        return None


class DocumentResultSink(_BufferingSink):
    """
    After LoadAndParse: persist Document rows (upsert on source_id) and thread the
    stored id forward as ``metadata['doc_id']`` so chunks resolve ``parent_doc_id``.
    """

    async def _persist(self, results: list[Any]) -> None:
        docs = [Document(content=item.content, metadata=item.metadata) for item in results]
        doc_ids = await self._store(docs)  # one transaction for the whole batch
        for item, doc_id in zip(results, doc_ids):
            item.metadata["doc_id"] = doc_id

    async def _store(self, docs: list[Document]) -> list[str]:
        return await self.repo.store_documents(docs)  # a (re-)ingest replaces the document's chunks


class DocumentRefreshSink(DocumentResultSink):
    """
    After a document-phase stage that REWRITES the document text (CleanAndNormalize): re-persist the
    document, so ``documents.content`` stays the text the chunks' char spans index into (the extractor's
    raw text was stored by LoadAndParse, before cleaning). Only document-phase items — those carrying
    ``item.document`` — are written; a chunk-phase item (a pipeline that cleans after chunking) passes
    through untouched, so it can never be mistaken for a document. The write is a row-only refresh, so a
    replayed job (at-least-once redelivery) leaves the chunks and vectors built after it in place.
    """

    async def _persist(self, results: list[Any]) -> None:
        documents = [item for item in results if item.document is not None]
        if documents:
            await super()._persist(documents)

    async def _store(self, docs: list[Document]) -> list[str]:
        return await self.repo.refresh_documents(docs)


class ChunkResultSink(_BufferingSink):
    """
    After Chunk: persist Chunk rows atomically and thread each stored id forward as
    ``metadata['chunk_id']`` for the downstream Enrich/Embed stages.
    """

    async def _persist(self, results: list[Any]) -> None:
        if not results:
            return
        chunks = [
            Chunk(
                parent_doc_id=item.metadata["doc_id"],
                content=item.content,
                chunk_index=item.metadata["chunk_index"],
                provenance=item.provenance,  # geometry / header_path / tree / table → persisted columns
                metadata=item.metadata,
            )
            for item in results
        ]
        chunk_ids = await self.repo.store_chunks(chunks)
        for item, chunk_id in zip(results, chunk_ids):
            item.metadata["chunk_id"] = chunk_id


class EmbeddingResultSink(_BufferingSink):
    """
    After Embed: bulk-persist Embedding results in persistence-batch-sized writes
    (independent of the worker's compute batch, D4).
    """

    def __init__(self, repository: ChunkStore, persist_batch_size: int = 128):
        super().__init__(repository)
        self.persist_batch_size = persist_batch_size

    async def _persist(self, results: list[Any]) -> None:
        for i in range(0, len(results), self.persist_batch_size):
            await self.repo.store_embeddings(results[i : i + self.persist_batch_size])


def create_sink_registry() -> dict[str, type[ResultSink]]:
    """Map each stage type tag to the ResultSink that persists its output. Keys MUST match
    stage ``.tag`` values; ``PipelineOrchestrator.begin_batch`` looks them up."""
    return {
        "LoadAndParse": DocumentResultSink,
        "Enrich": PassthroughSink,  # doc-phase enrichment annotates item.document; nothing to persist here
        "CleanAndNormalize": DocumentRefreshSink,  # cleaned text replaces the raw extraction in documents
        "Chunk": ChunkResultSink,
        "Embed": EmbeddingResultSink,
    }
