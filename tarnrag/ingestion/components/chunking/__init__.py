"""Chunking: split a loaded document into retrieval chunks + the auto-merging tree.

Importing this package registers the ``Chunk`` stage and the built-in chunkers (``structure_aware``,
``recursive``) with the global ``ComponentFactory``, so ``Pipeline.from_spec`` / ``ComponentFactory``
can build them. The chunkers are imported before ``ChunkStage`` (the driver builds a chunker child).
"""

from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.chunking.chunker import Chunker
from tarnrag.ingestion.components.chunking.recursive import RecursiveCharacterChunker
from tarnrag.ingestion.components.chunking.structure_aware import StructureAwareChunker

__all__ = [
    "ChunkStage",
    "Chunker",  # TempChunk is reached as Chunker.TempChunk
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
]
