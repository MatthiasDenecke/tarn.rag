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
from tarnrag.contracts.ports import ChunkStore, DocumentFactsSource, JobStatusSource, RetrievalStore
from tarnrag.contracts.results import Candidate, ChunkFilter, ChunkRecord, RetrievalResult
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
    is_retrieval_unit,
)

__all__ = [
    # dtos
    "PipelineItem",
    "Document",
    "Chunk",
    "Embedding",
    "DocumentFacts",
    "CorpusStatus",
    "MethodRef",
    # structure (layout-aware extraction)
    "StructuredDocument",
    "Element",
    "ElementKind",
    "Table",
    "TableCell",
    "Annotation",
    "Span",
    "PageBox",
    "Geometry",
    "ChunkProvenance",
    "is_retrieval_unit",
    # ports
    "ChunkStore",
    "RetrievalStore",
    "DocumentFactsSource",
    "JobStatusSource",
    # results
    "Candidate",
    "ChunkFilter",
    "ChunkRecord",
    "RetrievalResult",
    # index_meta
    "IndexMeta",
]
