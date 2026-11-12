"""The Docling PDF extractor. The DoclingDocument → StructuredDocument MAPPING is verified against a
constructed DoclingDocument (needs only ``docling-core``); the end-to-end converter path is gated on
the heavy ``docling`` package being installed."""

import importlib.util
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tarnrag.contracts import ElementKind, StructuredDocument
from tarnrag.ingestion.components.extraction import Source
from tarnrag.ingestion.components.extraction.docling_pdf import DoclingExtractor

requires_docling_core = pytest.mark.skipif(
    importlib.util.find_spec("docling_core") is None, reason="docling-core not installed"
)


def _prov(page, box, span=(0, 0)):
    from docling_core.types.doc.base import CoordOrigin
    from docling_core.types.doc.document import BoundingBox, ProvenanceItem

    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
        charspan=span,
    )


@requires_docling_core
def test_mapping_headings_paragraph_geometry_and_header_path():
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    d = DoclingDocument(name="t")
    d.add_title(text="Pump Manual", prov=_prov(1, (10, 20, 200, 35)))
    h = d.add_heading(text="Safety", level=1, prov=_prov(1, (10, 40, 80, 55)))
    d.add_text(label=DocItemLabel.TEXT, text="Wear PPE.", parent=h, prov=_prov(1, (10, 60, 150, 75)))

    doc = DoclingExtractor._map(d, Source(source_id="d1", source_kind="pdf"))
    assert isinstance(doc, StructuredDocument) and doc.extractor == "docling"
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.HEADING, "Pump Manual") in kinds
    assert (ElementKind.HEADING, "Safety") in kinds
    assert (ElementKind.PARAGRAPH, "Wear PPE.") in kinds

    para = next(e for e in doc.elements if e.text == "Wear PPE.")
    assert para.header_path == ["Pump Manual", "Safety"]  # title (level 0) -> heading (level 1)
    box = para.geometry[0].boxes[0]
    assert box.page == 1 and box.bbox[2] > box.bbox[0]  # page + a real bbox
    s = para.geometry[0]
    assert doc.text[s.start:s.end] == "Wear PPE."  # char offsets index doc.text exactly


@requires_docling_core
def test_mapping_table_cells_geometry_and_header_addressing():
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    from docling_core.types.doc.document import (
        DoclingDocument,
        ProvenanceItem,
        TableCell,
        TableData,
    )

    def bb(box):
        return BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT)

    def cell(text, r, c, **kw):
        return TableCell(
            text=text, start_row_offset_idx=r, end_row_offset_idx=r + 1,
            start_col_offset_idx=c, end_col_offset_idx=c + 1, row_span=1, col_span=1, **kw,
        )

    data = TableData(num_rows=2, num_cols=2, table_cells=[
        cell("Bolt", 0, 0, column_header=True, bbox=bb((0, 0, 50, 10))),
        cell("Torque", 0, 1, column_header=True, bbox=bb((50, 0, 100, 10))),
        cell("M6", 1, 0, row_header=True, bbox=bb((0, 10, 50, 20))),
        cell("6 Nm", 1, 1, bbox=bb((50, 10, 100, 20))),
    ])
    d = DoclingDocument(name="t")
    d.add_table(data=data, prov=ProvenanceItem(
        page_no=3, bbox=bb((0, 0, 100, 20)), charspan=(0, 0)))

    doc = DoclingExtractor._map(d, Source(source_id="d1", source_kind="pdf"))
    table_el = next(e for e in doc.elements if e.kind == ElementKind.TABLE)
    t = table_el.table
    assert t is not None
    data_cell = t.cell_at(1, 1)
    assert data_cell.text == "6 Nm"
    assert data_cell.geometry[0].boxes[0].page == 3  # cell-level geometry off the table's page
    assert [c.text for c in t.column_headers_for(data_cell)] == ["Torque"]  # address by column header
    assert [c.text for c in t.row_headers_for(data_cell)] == ["M6"]  # address by row header
    s = data_cell.geometry[0]
    assert doc.text[s.start:s.end] == "6 Nm"  # cell char-offset indexes doc.text exactly


@requires_docling_core
def test_mapping_lists_code_caption_and_skips_empty():
    """_map covers the remaining text-item kinds — list items, code, captions — and skips text-empty
    items (e.g. pictures); offsets still index doc.text exactly."""
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    d = DoclingDocument(name="t")
    d.add_list_item(text="Helmet", prov=_prov(1, (10, 10, 80, 20)))
    d.add_code(text="print('hi')", prov=_prov(1, (10, 25, 120, 40)))
    d.add_text(label=DocItemLabel.CAPTION, text="Figure 1: the pump.", prov=_prov(1, (10, 45, 120, 55)))
    d.add_picture(prov=_prov(1, (10, 60, 60, 110)))  # no text -> skipped by _map

    doc = DoclingExtractor._map(d, Source(source_id="d1", source_kind="pdf"))
    kinds = {(e.kind, e.text) for e in doc.elements}
    assert (ElementKind.LIST_ITEM, "Helmet") in kinds
    assert (ElementKind.CODE, "print('hi')") in kinds
    assert (ElementKind.CAPTION, "Figure 1: the pump.") in kinds
    assert all(e.text.strip() for e in doc.elements)  # the text-less picture item was dropped
    for e in doc.elements:
        s = e.geometry[0]
        assert doc.text[s.start:s.end] == e.text  # char offsets index doc.text exactly


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None, reason="docling (the converter) not installed"
)
def test_docling_end_to_end_on_fixture():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"
    doc = DoclingExtractor(DoclingExtractor.Config()).extract(
        Source(source_id="d1", source_kind="pdf", path=str(fixture))
    )
    assert doc.extractor == "docling" and doc.elements


def test_docling_extractor_requires_a_path():
    with pytest.raises(ValueError):
        DoclingExtractor(DoclingExtractor.Config()).extract(Source(source_id="d1", content="x"))


def test_concurrent_extractions_share_one_converter_and_take_turns(monkeypatch):
    """A distributed worker runs up to WORKER__CONCURRENCY batches in threads at once. Concurrent first
    extractions build the converter (and its models) once, and conversions on it never overlap, since
    docling doesn't promise that concurrent ``convert`` calls on one converter are safe."""
    built, running, peak = [], [0], [0]
    count = threading.Lock()

    class FakeConverter:  # stands in for docling's DocumentConverter (not installed in the test env)
        def __init__(self):
            built.append(self)
            time.sleep(0.05)  # a model load: long enough for every thread to reach the check

        def convert(self, path):
            with count:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with count:
                running[0] -= 1
            return types.SimpleNamespace(document=path)

    module = types.ModuleType("docling.document_converter")
    module.DocumentConverter = FakeConverter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)
    monkeypatch.setattr(DoclingExtractor, "_map", staticmethod(lambda dl_doc, source: dl_doc))
    extractor = DoclingExtractor(DoclingExtractor.Config())
    start = threading.Barrier(4)

    def extract(i):
        start.wait()
        return extractor.extract(Source(source_id=f"d{i}", source_kind="pdf", path=f"/docs/{i}.pdf"))

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(extract, range(4))) == [f"/docs/{i}.pdf" for i in range(4)]
    assert len(built) == 1 and peak[0] == 1
