"""Cross-boundary contracts — the shared kernel.

The DTOs, ports, retrieval-result types, and the ``index_meta`` build record that ingestion,
storage, and retrieval all depend on — and which depends on none of them (a leaf package). Import
from here: ``from tarnrag.contracts import Document, ChunkStore, Candidate, RetrievalResult``.
"""

from tarnrag.contracts.dtos import (
    Chunk,
    CorpusStatus,
    Document,
    DocumentFacts,
    Embedding,
    MethodRef,
    PipelineItem,
)
from tarnrag.contracts.index_meta import IndexMeta
from tarnrag.contracts.ports import (
    ChunkStore,
    DocumentFactsSource,
    JobStatusSource,
    RetrievalStore,
)
from tarnrag.contracts.results import (
    Candidate,
    ChunkFilter,
    ChunkRecord,
    RetrievalResult,
)
from tarnrag.contracts.structure import (
    Annotation,
    ChunkProvenance,
    Element,
    ElementKind,
    Geometry,
    PageBox,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)

__all__ = [
    "Annotation",
    "Candidate",
    "Chunk",
    "ChunkFilter",
    "ChunkProvenance",
    "ChunkRecord",
    "ChunkStore",
    "CorpusStatus",
    "Document",
    "DocumentFacts",
    "DocumentFactsSource",
    "Element",
    "ElementKind",
    "Embedding",
    "Geometry",
    "IndexMeta",
    "JobStatusSource",
    "MethodRef",
    "PageBox",
    "PipelineItem",
    "RetrievalResult",
    "RetrievalStore",
    "Span",
    "StructuredDocument",
    "Table",
    "TableCell",
]
