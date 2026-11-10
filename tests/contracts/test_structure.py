"""The layout-aware extraction contracts: the StructuredDocument/table model and the PipelineItem
extension. Pure data + helper tests — no engine, no I/O."""

import pytest
from pydantic import ValidationError

from tarnrag.contracts import (
    Annotation,
    ChunkProvenance,
    Element,
    ElementKind,
    PageBox,
    PipelineItem,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)

# ---- geometry: offsets universal, boxes additive ----

def test_span_offsets_are_universal_boxes_optional():
    text_span = Span(start=0, end=10)
    assert text_span.boxes == []  # text/markdown: offsets only
    pdf_span = Span(start=0, end=10, boxes=[PageBox(page=1, bbox=(0, 0, 100, 20))])
    assert pdf_span.boxes[0].page == 1
    # both kinds share the SAME offset substrate — so offset-based ops work on either
    assert (text_span.start, text_span.end) == (pdf_span.start, pdf_span.end)


def test_span_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Span(start=10, end=5)


def test_pagebox_page_is_one_based():
    with pytest.raises(ValidationError):
        PageBox(page=0, bbox=(0, 0, 1, 1))


# ---- table addressing (FR-3.3 / AC-3) ----

def _grid_table() -> Table:
    # columns: "Bolt" (col 0) / "Torque" (col 1); header row 0; one data row.
    return Table(
        n_rows=2,
        n_cols=2,
        cells=[
            TableCell(id="h0", row=0, col=0, is_column_header=True, text="Bolt"),
            TableCell(id="h1", row=0, col=1, is_column_header=True, text="Torque"),
            TableCell(id="r1c0", row=1, col=0, is_row_header=True, text="M6"),
            TableCell(
                id="r1c1", row=1, col=1, text="6 Nm",
                geometry=[Span(start=40, end=44, boxes=[PageBox(page=2, bbox=(1, 2, 3, 4))])],
            ),
        ],
        markdown="| Bolt | Torque |\n|---|---|\n| M6 | 6 Nm |",
    )


def test_cell_addressing_by_id_and_header():
    t = _grid_table()
    cell = t.cell_at(1, 1)
    assert cell is not None and cell.id == "r1c1"
    assert cell.geometry[0].boxes[0].page == 2  # cell-level geometry (FR-3.2)
    assert [c.text for c in t.column_headers_for(cell)] == ["Torque"]  # query by column header
    assert [c.text for c in t.row_headers_for(cell)] == ["M6"]  # query by row header


def test_cell_at_honours_spans():
    t = Table(
        n_rows=1, n_cols=3,
        cells=[
            TableCell(id="wide", row=0, col=0, col_span=2, text="merged"),
            TableCell(id="c2", row=0, col=2, text="x"),
        ],
    )
    assert t.cell_at(0, 0).id == "wide"
    assert t.cell_at(0, 1).id == "wide"  # covered by the col_span
    assert t.cell_at(0, 2).id == "c2"
    assert t.cell_at(0, 9) is None


# ---- structured document ----

def test_structured_document_lookup_and_defaults():
    doc = StructuredDocument(
        source_id="d1", source_kind="markdown", extractor="markdown",
        text="# Title\n\nHello world.",
        elements=[
            Element(id="e0", kind=ElementKind.HEADING, text="Title", level=1,
                    geometry=[Span(start=0, end=7)]),
            Element(id="e1", kind=ElementKind.PARAGRAPH, text="Hello world.",
                    header_path=["Title"], geometry=[Span(start=9, end=21)]),
        ],
        content_hash="abc",
    )
    assert doc.element_by_id("e1").header_path == ["Title"]
    assert doc.element_by_id("missing") is None
    assert doc.elements[1].atomic is True  # FR-7.2 default


def test_annotations_attach_with_provenance_and_determinism_flag():
    el = Element(id="e1", kind=ElementKind.PARAGRAPH, text="Acme ships pumps.")
    el.annotations.append(
        Annotation(producer="ner-stub", type="entity", value={"label": "ORG", "text": "Acme"},
                   span=[Span(start=0, end=4)], deterministic=True)
    )
    assert el.annotations[0].producer == "ner-stub"
    assert el.annotations[0].deterministic is True


def test_element_subspan_maps_local_offsets_into_the_document():
    el = Element(id="e1", kind=ElementKind.PARAGRAPH, text="Wear PPE.", geometry=[Span(start=40, end=49)])
    s = el.subspan(5, 8)  # "PPE" within the element, offset by the element's document position
    assert (s.start, s.end) == (45, 48)
    el0 = Element(id="e0", kind=ElementKind.PARAGRAPH, text="x")  # no geometry -> base 0
    assert (el0.subspan(0, 1).start, el0.subspan(0, 1).end) == (0, 1)


# ---- PipelineItem extension (FR-6.1) ----

def test_pipelineitem_is_backward_compatible():
    item = PipelineItem(content="hello")
    assert item.document is None and item.provenance is None  # existing callers unaffected


def test_pipelineitem_carries_document_then_provenance():
    doc = StructuredDocument(source_id="d1", source_kind="pdf", extractor="docling",
                             text="body", content_hash="h")
    doc_phase = PipelineItem(content=doc.text, document=doc)
    assert doc_phase.document.extractor == "docling"

    chunk_phase = PipelineItem(
        content="body",
        provenance=ChunkProvenance(source_element_ids=["e0"], content_hash="h",
                                   geometry=[Span(start=0, end=4)], header_path=["Intro"]),
    )
    assert chunk_phase.provenance.source_element_ids == ["e0"]
    assert chunk_phase.document is None


def test_pipelineitem_round_trips_through_json():
    # PipelineItem is the inline job payload — the structured payload must serialize losslessly.
    doc = StructuredDocument(
        source_id="d1", source_kind="pdf", extractor="docling", text="body", content_hash="h",
        elements=[Element(id="e0", kind=ElementKind.PARAGRAPH, text="body",
                          geometry=[Span(start=0, end=4, boxes=[PageBox(page=1, bbox=(0, 0, 1, 1))])])],
    )
    item = PipelineItem(content="body", document=doc)
    restored = PipelineItem.model_validate_json(item.model_dump_json())
    assert restored.document.elements[0].geometry[0].boxes[0].page == 1
    assert restored.document.content_hash == "h"


def test_table_contextual_text_binds_values_to_headers():
    """The embedding-friendly rendering (P1): one line per data row, every value bound to its row and
    column headers — so 'goodwill in 2019' and '1,910' share one breath."""
    from tarnrag.contracts import Table, TableCell

    table = Table(
        n_rows=3, n_cols=3,
        cells=[
            TableCell(id="h1", row=0, col=1, is_column_header=True, text="2019"),
            TableCell(id="h2", row=0, col=2, is_column_header=True, text="2018"),
            TableCell(id="r1", row=1, col=0, is_row_header=True, text="Goodwill"),
            TableCell(id="v1", row=1, col=1, text="1,910"),
            TableCell(id="v2", row=1, col=2, text="2,130"),
            TableCell(id="r2", row=2, col=0, is_row_header=True, text="Revenue"),
            TableCell(id="v3", row=2, col=1, text="10"),
            TableCell(id="v4", row=2, col=2, text="8"),
        ],
    )
    assert table.contextual_text() == (
        "Goodwill \u2014 2019: 1,910; 2018: 2,130\nRevenue \u2014 2019: 10; 2018: 8"
    )


def test_table_contextual_text_without_headers_or_cells():
    from tarnrag.contracts import Table, TableCell

    bare = Table(n_rows=1, n_cols=2, cells=[
        TableCell(id="a", row=0, col=0, text="x"), TableCell(id="b", row=0, col=1, text="y"),
    ])
    assert bare.contextual_text() == "x; y"  # no headers to bind -> plain values, one row-line
    assert Table(n_rows=0, n_cols=0, cells=[]).contextual_text() == ""  # no data cells
