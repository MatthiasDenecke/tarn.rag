"""The Chunker component family: a ``StructuredDocument`` → retrieval chunks.

A ``Chunker`` is a pure ``Component`` (document in, chunks out); the ``ChunkStage`` driver builds the
configured chunker and threads each chunk's ``ChunkProvenance`` onto the chunk-phase ``PipelineItem``.
This is the same shape extraction uses — an ``Extractor`` does the work, ``LoadAndParse`` drives it —
and being a pure function makes a chunker unit-testable without the pipeline.

``Chunker.TempChunk`` is the *in-flight* emission unit (text + provenance + a positional parent link).
The ``Temp*`` prefix is the convention for a pipeline-phase, pre-persistence representation, distinct
from the at-rest ``tarnrag.contracts.Chunk`` DTO (built only at storage time).
"""

from __future__ import annotations

from abc import abstractmethod

from bausatz import Component
from pydantic import BaseModel

from tarnrag.contracts import (
    Annotation,
    ChunkProvenance,
    Element,
    Span,
    StructuredDocument,
)
from tarnrag.core.hashing import compute_content_hash


class Chunker(Component):
    """Port: turn a ``StructuredDocument`` into an ordered list of ``TempChunk``s — parents before
    their children, so ``parent_index`` always points backward."""

    class TempChunk(BaseModel):
        """The in-flight chunk a ``Chunker`` emits (``Temp*`` = a pre-persistence, pipeline-phase
        representation): its ``text``, the ``provenance`` linking it to source elements, and — for the
        auto-merging tree — ``parent_index``, the slot of this chunk's section parent in the returned
        list (the driver resolves it to a ``parent_chunk_id`` at persist time). Distinct from the
        at-rest ``tarnrag.contracts.Chunk`` DTO."""

        text: str
        provenance: ChunkProvenance
        parent_index: int | None = None  # parent chunk's index in the same list; None ⇒ a root/leaf w/o parent

    class Config(Component.Config):
        """Base chunker config; concrete chunkers pin ``class_name`` and add their own options."""

    @abstractmethod
    def chunk(self, document: StructuredDocument) -> list[Chunker.TempChunk]:
        """The document's chunks (parents before children); at least one for non-empty text."""

    @staticmethod
    def _hash(text: str) -> str:
        """The chunk content hash — the dedup/identity key."""
        return compute_content_hash(text)

    @staticmethod
    def _union_geometry(elements: list[Element]) -> list[Span]:
        """All the elements' spans, concatenated in order — the chunk's aggregated geometry (the
        per-span page boxes ride along, so PDF highlighting survives the merge)."""
        return [span for element in elements for span in element.geometry]

    @staticmethod
    def _annotations(elements: list[Element]) -> list[Annotation]:
        """The annotations carried by the covered elements (entities/topics an enricher attached)."""
        return [annotation for element in elements for annotation in element.annotations]
