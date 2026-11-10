"""Docling PDF extractor (the high-fidelity tier) — DoclingDocument → StructuredDocument.

Docling's layout + table models give a structured document (heading hierarchy, tables with cell
geometry, reading order, page+bbox provenance); ``_map`` maps that onto our tool-neutral model. The
heavy converter (the ``docling`` package) is imported lazily and is the optional ``[docling]`` extra;
``_map`` itself depends only on ``docling-core``'s data model, so it is unit-tested against a
constructed ``DoclingDocument`` without running the converter.

Geometry: char offsets index OUR concatenated normalized text (consistent with the other extractors);
page+bbox come from Docling provenance (normalized to top-left when the page size is known). Pictures
without text are skipped for now (text-only normalized text).
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal

from tarnrag.contracts import (
    Element,
    ElementKind,
    PageBox,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source

_SEP = "\n\n"  # joins element texts into StructuredDocument.text


class DoclingExtractor(Extractor):
    """High-fidelity PDF/structured extraction via Docling (headings, tables, page+bbox geometry)."""

    class Config(Extractor.Config):
        class_name: Literal["docling"] = "docling"

    config: DoclingExtractor.Config

    def __init__(self, config: DoclingExtractor.Config) -> None:
        super().__init__(config)
        self._conv = None  # the DocumentConverter — built + cached on first use (models load once)

    def extract(self, source: Source) -> StructuredDocument:
        if not source.path:
            raise ValueError("the 'docling' extractor needs a document file path (source.path)")
        return self._map(self._converter().convert(source.path).document, source)

    def _converter(self):
        if self._conv is None:
            from docling.document_converter import (
                DocumentConverter,  # heavy; lazily imported once
            )

            self._conv = DocumentConverter()
        return self._conv

    # ------------------------------------------------------------------ mapping (docling-core only)
    @staticmethod
    def _map(dl_doc: Any, source: Source) -> StructuredDocument:
        """``DoclingDocument`` → ``StructuredDocument``. Pure (docling-core), so it is unit-testable
        without the converter."""
        from docling_core.types.doc.document import (
            ListItem,
            SectionHeaderItem,
            TableItem,
            TitleItem,
        )

        try:
            from docling_core.types.doc.document import CodeItem
        except ImportError:  # older docling-core
            CodeItem = ()  # type: ignore[assignment]

        elements: list[Element] = []
        parts: list[str] = []
        offset = 0
        stack: list[tuple[int, str, str]] = []  # (level, text, element_id) of open headings

        for idx, (item, _depth) in enumerate(dl_doc.iterate_items()):
            eid = f"e{idx}"
            table: Table | None = None
            if isinstance(item, TableItem):
                start = offset
                table, text = DoclingExtractor._table(item, dl_doc, base=start)
                kind, level = ElementKind.TABLE, None
            else:
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue  # skip empty / picture items (no text to chunk; v1 is text-only)
                start = offset
                kind, level = DoclingExtractor._kind(item, TitleItem, SectionHeaderItem, ListItem, CodeItem)

            end = start + len(text)
            offset = end + len(_SEP)
            parts.append(text)
            boxes = [
                PageBox(page=p.page_no, bbox=DoclingExtractor._bbox(p.bbox, dl_doc, p.page_no))
                for p in (getattr(item, "prov", None) or [])
            ]

            if kind == ElementKind.HEADING and level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                elements.append(Element(
                    id=eid, kind=kind, text=text, level=level,
                    header_path=[t for _, t, _ in stack],
                    parent_id=stack[-1][2] if stack else None,
                    geometry=[Span(start=start, end=end, boxes=boxes)],
                ))
                stack.append((level, text, eid))
            else:
                elements.append(Element(
                    id=eid, kind=kind, text=text,
                    header_path=[t for _, t, _ in stack],
                    parent_id=stack[-1][2] if stack else None,
                    geometry=[Span(start=start, end=end, boxes=boxes)],
                    table=table,
                ))

        doc_text = _SEP.join(parts)
        return DoclingExtractor._create_document(source, doc_text, elements, extractor="docling", default_kind="pdf")

    @staticmethod
    def _kind(item, TitleItem, SectionHeaderItem, ListItem, CodeItem):
        """Map a Docling text item to (ElementKind, heading level | None)."""
        if isinstance(item, TitleItem):
            return ElementKind.HEADING, 0
        if isinstance(item, SectionHeaderItem):
            return ElementKind.HEADING, int(getattr(item, "level", 1) or 1)
        if isinstance(item, ListItem):
            return ElementKind.LIST_ITEM, None
        if CodeItem and isinstance(item, CodeItem):
            return ElementKind.CODE, None
        if getattr(getattr(item, "label", None), "value", "") == "caption":
            return ElementKind.CAPTION, None
        return ElementKind.PARAGRAPH, None

    @staticmethod
    def _table(item, dl_doc, base: int) -> tuple[Table, str]:
        """Build a ``Table`` (cells with char-offset + cell bbox geometry) and its rendered text. Cell
        char spans are absolute (offset by ``base`` — the table's start in the normalized text)."""
        data = item.data
        by_row: dict[int, list] = {}
        for c in (getattr(data, "table_cells", None) or []):
            by_row.setdefault(c.start_row_offset_idx, []).append(c)
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else None

        out_cells: list[TableCell] = []
        buf: list[str] = []
        length = 0

        def emit(s: str) -> int:
            nonlocal length
            pos = length
            buf.append(s)
            length += len(s)
            return pos

        for r in sorted(by_row):
            for j, c in enumerate(sorted(by_row[r], key=lambda x: x.start_col_offset_idx)):
                if j > 0:
                    emit(" | ")
                ctext = (c.text or "").strip()
                cstart = base + emit(ctext)
                cell_boxes = (
                    [PageBox(page=page_no, bbox=DoclingExtractor._bbox(c.bbox, dl_doc, page_no))]
                    if (getattr(c, "bbox", None) is not None and page_no is not None)
                    else []
                )
                out_cells.append(TableCell(
                    id=f"r{c.start_row_offset_idx}c{c.start_col_offset_idx}",
                    row=c.start_row_offset_idx,
                    col=c.start_col_offset_idx,
                    row_span=max(1, (c.end_row_offset_idx or c.start_row_offset_idx + 1) - c.start_row_offset_idx),
                    col_span=max(1, (c.end_col_offset_idx or c.start_col_offset_idx + 1) - c.start_col_offset_idx),
                    is_column_header=bool(getattr(c, "column_header", False)),
                    is_row_header=bool(getattr(c, "row_header", False)),
                    text=ctext,
                    geometry=[Span(start=cstart, end=cstart + len(ctext), boxes=cell_boxes)],
                ))
            emit("\n")

        n_rows = int(getattr(data, "num_rows", 0) or (max(by_row) + 1 if by_row else 0))
        n_cols = int(getattr(data, "num_cols", 0) or 0)
        return Table(n_rows=n_rows, n_cols=n_cols, cells=out_cells, markdown="".join(buf)), "".join(buf)

    @staticmethod
    def _bbox(bbox, dl_doc, page_no) -> tuple[float, float, float, float]:
        """A Docling ``BoundingBox`` → ``(x0, y0, x1, y1)`` tuple, normalized to top-left origin when
        the page height is known (so PDF geometry is consistent across extractors)."""
        pages = getattr(dl_doc, "pages", None)
        size = getattr(pages.get(page_no), "size", None) if pages and page_no in pages else None
        height = getattr(size, "height", None) if size else None
        if height is not None:
            with contextlib.suppress(Exception):  # geometry is best-effort; fall back to raw coords
                bbox = bbox.to_top_left_origin(height)
        return tuple(round(float(v), 2) for v in bbox.as_tuple())
