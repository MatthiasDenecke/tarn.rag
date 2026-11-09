"""ChunkStage — the chunking driver: build the configured ``Chunker`` and emit chunk-phase items.

Like ``LoadAndParse`` drives the extractors, ``ChunkStage`` is a container that builds one ``Chunker``
child (default ``structure_aware``) from config and runs it over ``item.document``, threading each
chunk's ``ChunkProvenance`` onto the produced chunk-phase ``PipelineItem``. The positional parent link
and the chunk ``level`` ride forward in metadata so the persistence slice can resolve the tree. When no
document is present (plain-text ingest with no prior extract), the text is wrapped as a one-paragraph
document so chunking still runs.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Literal

from pydantic import Field

from tarnrag.contracts import Element, ElementKind, PipelineItem, Span, StructuredDocument
from bausatz import ComponentFactory
from tarnrag.core.hashing import compute_content_hash
from tarnrag.ingestion.components.chunking.chunker import Chunker
from tarnrag.ingestion.components.chunking.structure_aware import StructureAwareChunker
from tarnrag.ingestion.pipeline.pipeline import PipelineStage


class ChunkStage(PipelineStage):
    """Drive the configured ``Chunker`` over ``item.document``, emitting one chunk-phase item per chunk."""

    class Config(PipelineStage.Config):
        class_name: Literal["Chunk"] = "Chunk"
        # The chunker child (a spec's ``class_name`` + options, validated into its concrete Config);
        # config-driven, like LoadAndParse's routes.
        chunker: Chunker.Config = Field(default_factory=StructureAwareChunker.Config)

    config: ChunkStage.Config

    def __init__(self, config: ChunkStage.Config) -> None:
        super().__init__(config)
        self._chunker: Chunker | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the chunker child through the framework factory (the hook containers override)."""
        self._chunker = factory.build_as(self.config.chunker, Chunker)

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        self.ensure_built()
        document = item.document or self._synthesize(item)
        chunks = self._chunker.chunk(document)
        if not chunks:
            raise ValueError(f"{self.name} produced no chunks for {item.metadata.get('doc_id')}")
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            yield PipelineItem(
                content=chunk.text,
                metadata={
                    **item.metadata,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "chunk_level": chunk.provenance.level,
                    "parent_ordinal": chunk.parent_index,  # resolved to parent_chunk_id at persist
                },
                provenance=chunk.provenance,
            )

    @staticmethod
    def _synthesize(item: PipelineItem) -> StructuredDocument:
        """Wrap raw ``item.content`` as a single-paragraph document, so chunking runs even when no
        extractor populated ``item.document`` (e.g. plain-text ingest)."""
        text = item.content or ""
        md = item.metadata
        return StructuredDocument(
            source_id=md.get("doc_id") or md.get("source_id") or str(uuid.uuid4()),
            source_kind=md.get("source_type") or "text",
            extractor="none",
            text=text,
            elements=(
                [Element(id="e0", kind=ElementKind.PARAGRAPH, text=text, geometry=[Span(start=0, end=len(text))])]
                if text
                else []
            ),
            content_hash=compute_content_hash(text),
        )
