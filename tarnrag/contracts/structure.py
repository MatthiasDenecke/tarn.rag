"""Structured-document contracts — the layout-aware extraction model.

Produced by a format extractor (PDF/Markdown/…) and refined by enrichment components, a
``StructuredDocument`` is the rich, in-flight representation of one ingested document: an ordered
list of ``Element``s (reading order) with hierarchy, per-element geometry, header paths, and
extensible enricher ``Annotation``s. It is the *document-phase* payload on ``PipelineItem`` (the
*chunk-phase* carries ``ChunkProvenance`` instead).

Geometry decomposition (the load-bearing decision): **char offsets are universal, page boxes are
additive.** Every ``Span`` carries ``start``/``end`` offsets into the document's normalized ``text``
(the substrate that gets chunked and embedded), so all offset-based work (overlap, text-view
highlighting, late chunking, sub-element annotation spans) is uniform across PDF and text; a span
*additionally* carries page ``boxes`` when the source is paged/visual (PDF).

This module is contracts only — no extractor, enricher, chunker, or schema (see
``doc/structured-document-design.md`` and ``doc/layout-aware-extraction-requirements.md``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PageBox(BaseModel):
    """A bounding box on one rendered page (PDF). ``bbox`` is ``(x0, y0, x1, y1)`` in the
    extractor's coordinate space, interpreted consistently per ``source_kind``."""

    page: int = Field(ge=1)  # 1-based page number
    bbox: tuple[float, float, float, float]


class Span(BaseModel):
    """A contiguous region of a document, addressed by **char offsets into the normalized text**
    (universal) plus optional **page boxes** (PDF visual highlighting; empty for text/markdown)."""

    start: int = Field(ge=0)  # char offset into StructuredDocument.text
    end: int = Field(ge=0)
    boxes: list[PageBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")
        return self


# An element's or chunk's location: usually one contiguous span; a list when a chunk merges
# non-adjacent source (e.g. an injected header path + a body paragraph).
Geometry = list[Span]


class ElementKind(str, Enum):
    """The structural kind of an element. Extractors map their own labels onto this set."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"


class Annotation(BaseModel):
    """An enricher's finding on a document or element (or a sub-span of one). Records its
    ``producer`` and a ``deterministic`` flag so generative findings are never silently trusted."""

    producer: str  # the enricher Component's name (or the base extractor, e.g. "docling")
    type: str  # "entity" | "topic" | "classification" | …
    value: dict[str, Any] = Field(default_factory=dict)
    span: Geometry | None = None  # sub-span(s) it applies to; None ⇒ the whole element/document
    deterministic: bool = True  # False ⇒ generated (must be constrained/verified, never asserted as fact)


class TableCell(BaseModel):
    """One cell of a ``Table``: grid position + spans, header flags (column/row), text, and
    cell-level geometry — enough to address a cell by id or resolve it to its headers."""

    id: str
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    is_column_header: bool = False
    is_row_header: bool = False
    text: str = ""
    geometry: Geometry = Field(default_factory=list)


class Table(BaseModel):
    """A structured table: ``cells`` (with spans, header flags, and cell geometry) plus a rendered
    ``markdown`` form for embedding/display. Helpers resolve a data cell to its header cells."""

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str = ""

    def cell_at(self, row: int, col: int) -> TableCell | None:
        """The cell occupying grid position ``(row, col)``, honouring row/col spans."""
        for c in self.cells:
            if c.row <= row < c.row + c.row_span and c.col <= col < c.col + c.col_span:
                return c
        return None

    def column_headers_for(self, cell: TableCell) -> list[TableCell]:
        """The column-header cells whose columns overlap ``cell`` (its column header(s))."""
        return [c for c in self.cells if c.is_column_header and self._cols_overlap(c, cell)]

    def row_headers_for(self, cell: TableCell) -> list[TableCell]:
        """The row-header cells whose rows overlap ``cell`` (its row header(s))."""
        return [c for c in self.cells if c.is_row_header and self._rows_overlap(c, cell)]

    def contextual_text(self) -> str:
        """The header-contextualized linearization — one line per data row, each value bound to its
        headers: ``{row header} — {column header}: {value}; …``. This is the *embedding-friendly*
        rendering (P1): a raw grid buries what a number means several lines from where it sits, so cell
        tokens embed weakly against a natural-language question; binding every value to its headers puts
        the question's vocabulary ("goodwill impairment in 2019") and the value in one breath.
        ``markdown`` remains the display / BM25 form. Empty when the table has no data cells."""
        data = [c for c in self.cells if not (c.is_column_header or c.is_row_header) and c.text.strip()]
        by_row: dict[int, list[TableCell]] = {}
        for cell in data:
            by_row.setdefault(cell.row, []).append(cell)
        lines: list[str] = []
        for row in sorted(by_row):
            cells = sorted(by_row[row], key=lambda c: c.col)
            row_label = " ".join(h.text.strip() for h in self.row_headers_for(cells[0]) if h.text.strip())
            pairs = []
            for cell in cells:
                col_label = " ".join(
                    h.text.strip() for h in self.column_headers_for(cell) if h.text.strip()
                )
                pairs.append(f"{col_label}: {cell.text.strip()}" if col_label else cell.text.strip())
            line = "; ".join(pairs)
            lines.append(f"{row_label} — {line}" if row_label else line)
        return "\n".join(lines)

    @staticmethod
    def _cols_overlap(a: TableCell, b: TableCell) -> bool:
        return a.col < b.col + b.col_span and b.col < a.col + a.col_span

    @staticmethod
    def _rows_overlap(a: TableCell, b: TableCell) -> bool:
        return a.row < b.row + b.row_span and b.row < a.row + a.row_span


class Element(BaseModel):
    """One structural unit of a document, in reading order. Hierarchy is by reference (``parent_id``
    + heading ``level``); ``header_path`` is precomputed so consumers never re-walk the tree."""

    id: str
    kind: ElementKind
    text: str = ""  # the element's canonical extracted text
    geometry: Geometry = Field(default_factory=list)
    header_path: list[str] = Field(default_factory=list)  # e.g. ["3 Safety", "3.2 Lockout"]
    parent_id: str | None = None  # enclosing section/heading element
    level: int | None = None  # heading depth, when kind == HEADING
    atomic: bool = True  # the chunker must not split this element (FR-7.2)
    sentences: list[Span] | None = None  # best-effort sentence sub-spans; None ⇒ not provided
    annotations: list[Annotation] = Field(default_factory=list)
    table: Table | None = None  # set iff kind == TABLE

    def subspan(self, start: int, end: int) -> Span:
        """The absolute document span of this element's local range ``text[start:end]`` — the element's
        char offset plus the range — so a sub-region indexes ``StructuredDocument.text`` like any Span."""
        base = self.geometry[0].start if self.geometry else 0
        return Span(start=base + start, end=base + end)


class StructuredDocument(BaseModel):
    """The document-phase, in-flight representation: ordered ``elements`` over a normalized ``text``
    substrate, with document-level ``annotations``. Mutable so enrichers can append annotations."""

    source_id: str
    source_kind: str  # "pdf" | "markdown" | …
    extractor: str  # which extractor produced this (provenance)
    text: str  # the normalized document text — the substrate Span offsets index into
    elements: list[Element] = Field(default_factory=list)  # reading order
    annotations: list[Annotation] = Field(default_factory=list)  # document-level
    content_hash: str

    def element_by_id(self, element_id: str) -> Element | None:
        """The element with this id, or ``None``."""
        return next((e for e in self.elements if e.id == element_id), None)


class ChunkProvenance(BaseModel):
    """The chunk-phase payload: a chunk's link back to the source elements it covers, the aggregated
    geometry/header-path needed to cite and highlight it, and its place in the auto-merging tree.
    Filled by the Chunk stage."""

    source_element_ids: list[str] = Field(default_factory=list)  # the (contiguous) source elements
    geometry: Geometry = Field(default_factory=list)  # aggregated across the merged elements
    header_path: list[str] = Field(default_factory=list)
    content_hash: str
    level: int = 0  # auto-merging tree: 0 = leaf chunk, 1 = section parent
    parent_chunk_id: str | None = None  # the section-parent's chunk id; resolved when chunks are persisted
    annotations: list[Annotation] = Field(default_factory=list)  # annotations overlapping the chunk
    table: Table | None = None  # set iff this chunk IS a table — carries cells/geometry for table_cells


def is_retrieval_unit(provenance: ChunkProvenance | None) -> bool:
    """Whether a chunk is searched — a leaf (``level`` 0, or a chunk with no tree provenance at all).
    A section parent (``level`` > 0) is stored so the auto-merger can hydrate it, but it is never
    embedded, sparse-indexed, or returned by a retriever: its text is its leaves' text again, so
    retrieving both would hand the reader the same passage twice. The single definition the embed stage,
    the sparse index, and the stores' ``level = 0`` search predicate agree on."""
    return provenance is None or provenance.level == 0
