"""EmbedStage — vectorize chunks into Embeddings (terminal stage).

Uses the shared ONNX ``OnnxEmbedder`` (passage side), so ingestion embeds with exactly the
pipeline retrieval will replay for queries (§5.3). The embedding identity lives on the stage's
``Config`` as an ``EmbeddingSettings`` (the factory copies ``settings.embedding`` in, so it stays
in sync with the retrieval-side embedder and its fingerprint). The embedder is built lazily via
``Embedder.create`` (local ONNX or an API backend, per ``embedding.provider``); tests inject a fake by
setting ``stage._embedder`` (anything with ``embed_passages(list[str]) -> list[list[float]]``).

When ``embedding.inject_header_path`` is set, each chunk's section header path is prepended to its
text before embedding (the chunk's structural context, from ``item.provenance.header_path``). It's an
embed-time index variant — queries are never injected — and rides the embedding fingerprint, so an
injected index won't ``open()`` with a non-injecting embedder; the two are compared as separate indexes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Literal

from pydantic import Field

from tarnrag.contracts import Embedding, PipelineItem, is_retrieval_unit
from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.ingestion.pipeline.pipeline import PipelineStage


class EmbedStage(PipelineStage):
    """
    Terminal stage: yields ``Embedding``s (not ``PipelineItem``s). ``process_batch`` groups items
    into ``embedding.batch_size``-sized embed calls; ``chunk_id`` comes from
    ``metadata['chunk_id']`` (set by ChunkResultSink). Only **retrieval units** are embedded: a section
    parent (``provenance.level > 0``) is stored for the auto-merger to hydrate, never searched, so it gets
    no vector (its text is its leaves' text again — embedding it would only crowd the index).
    """

    class Config(PipelineStage.Config):
        class_name: Literal["Embed"] = "Embed"
        embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
        embedding_dimension: int = 384

    config: EmbedStage.Config

    def __init__(self, config: EmbedStage.Config) -> None:
        super().__init__(config)
        self._embedder = None  # lazy; built on first use (or injected by tests)
        # A distributed worker runs up to WORKER__CONCURRENCY batches in threads at once, so its first Embed
        # batches race the lazy build — the lock makes them share one embedder (one copy of the model).
        self._embedder_lock = threading.Lock()

    def accepts(self, item: PipelineItem) -> bool:
        """Only retrieval units get a vector — the orchestrator never enqueues an Embed job for a section
        parent."""
        return is_retrieval_unit(item.provenance)

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        """
        Embed items in ``embedding.batch_size`` groups, yielding one ``Embedding`` per chunk
        (``chunk_id`` taken from ``metadata['chunk_id']``).
        """
        items = [it for it in items if self.accepts(it)]  # also when called directly, not via a job
        if not items:
            return
        embedder = self._get_embedder()
        embedding = self.config.embedding
        batch_size = embedding.batch_size
        for i in range(0, len(items), batch_size):
            sub = items[i : i + batch_size]
            vectors = embedder.embed_passages(
                [self._embed_text(it, embedding.inject_header_path, embedding.contextualize_tables) for it in sub]
            )
            for it, vec in zip(sub, vectors):
                # Just the chunk key + its vector: embedder identity is index-wide (index_meta).
                yield Embedding(chunk_id=it.metadata["chunk_id"], vector=list(vec))

    @staticmethod
    def _embed_text(item: PipelineItem, inject: bool, contextualize: bool) -> str:
        """The passage text to embed. Base: the chunk content — except a TABLE chunk under
        ``contextualize_tables`` (P1), which embeds its header-contextualized rendering (each value
        bound to its row/column headers) instead of the raw grid; the stored/BM25 text is untouched.
        ``inject_header_path`` then prefixes the section header path on top (the two compose). Both
        are no-ops when the item lacks the structure (the plain-text path), so content is always
        embedded."""
        text = item.content
        if contextualize and item.provenance and item.provenance.table:
            text = item.provenance.table.contextual_text() or item.content
        if inject and item.provenance and item.provenance.header_path:
            text = " > ".join(item.provenance.header_path) + "\n" + text
        return text

    def _get_embedder(self):
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:  # another thread may have built it while we waited
                    from tarnrag.core.resources.embedder import Embedder

                    self._embedder = Embedder.create(self.config.embedding, self.config.embedding_dimension)
        return self._embedder
