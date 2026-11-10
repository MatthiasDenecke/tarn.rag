"""The chunker family: the structure-aware tree (sections, atomic tables, auto-merging parents +
geometry/header-path threading), the recursive fallback's provenance, and the ChunkStage driver."""

from tarnrag.contracts import (
    Element,
    ElementKind,
    PageBox,
    PipelineItem,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.chunking.recursive import RecursiveCharacterChunker
from tarnrag.ingestion.components.chunking.structure_aware import StructureAwareChunker


def _doc(elements, text="x" * 400):
    return StructuredDocument(
        source_id="d1", source_kind="markdown", extractor="markdown",
        text=text, elements=elements, content_hash="h",
    )


def _el(eid, kind, text, header_path=(), boxes=(), start=0, end=10, **kw):
    span = Span(start=start, end=end, boxes=[PageBox(page=p, bbox=b) for p, b in boxes])
    return Element(id=eid, kind=kind, text=text, header_path=list(header_path), geometry=[span], **kw)


# --- structure-aware ---------------------------------------------------------- #

def test_section_with_many_leaves_emits_a_parent_then_its_leaves():
    els = [
        _el("h", ElementKind.HEADING, "Safety", level=1),
        _el("p1", ElementKind.PARAGRAPH, "Wear PPE.", header_path=["Safety"]),
        _el("p2", ElementKind.PARAGRAPH, "Lock out power.", header_path=["Safety"]),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config(max_chars=20)).chunk(_doc(els))

    levels = [c.provenance.level for c in chunks]
    assert levels == [1, 0, 0]  # the section parent precedes its two leaves
    parent, leaf1, leaf2 = chunks
    assert parent.parent_index is None and leaf1.parent_index == 0 and leaf2.parent_index == 0
    assert parent.provenance.source_element_ids == ["h", "p1", "p2"]  # parent covers the whole section
    assert [leaf1.provenance.source_element_ids, leaf2.provenance.source_element_ids] == [["p1"], ["p2"]]
    assert all(c.provenance.header_path == ["Safety"] for c in chunks)


def test_single_leaf_section_stays_flat():
    els = [
        _el("h", ElementKind.HEADING, "Intro", level=1),
        _el("p", ElementKind.PARAGRAPH, "A short intro.", header_path=["Intro"]),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config()).chunk(_doc(els))
    assert len(chunks) == 1  # nothing to merge up to — no parent
    assert chunks[0].provenance.level == 0 and chunks[0].parent_index is None
    assert chunks[0].provenance.header_path == ["Intro"]


def test_table_is_an_atomic_leaf_carrying_its_table():
    table = Table(n_rows=2, n_cols=2, markdown="| Bolt | Torque |", cells=[
        TableCell(id="c0", row=0, col=0, is_column_header=True, text="Bolt"),
        TableCell(id="c1", row=1, col=1, text="6 Nm"),
    ])
    els = [
        _el("h", ElementKind.HEADING, "Specs", level=1),
        _el("p", ElementKind.PARAGRAPH, "Torque values:", header_path=["Specs"]),
        _el("t", ElementKind.TABLE, "| Bolt | Torque |\n| M6 | 6 Nm |", header_path=["Specs"], table=table),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config()).chunk(_doc(els))
    table_chunk = next(c for c in chunks if c.provenance.table is not None)
    assert table_chunk.provenance.level == 0  # a leaf
    assert table_chunk.provenance.source_element_ids == ["t"]  # never merged with the paragraph
    assert table_chunk.provenance.table.cell_at(1, 1).text == "6 Nm"  # the Table rides on the provenance
    assert table_chunk.text.startswith("|")  # its markdown is the searchable text


def test_heading_only_section_emits_nothing_but_survives_in_header_path():
    # "Manual" (h1) has no direct body; its text lives on in the deeper section's header_path.
    els = [
        _el("h1", ElementKind.HEADING, "Manual", level=1),
        _el("h2", ElementKind.HEADING, "Safety", level=2, header_path=["Manual"]),
        _el("p", ElementKind.PARAGRAPH, "Wear PPE.", header_path=["Manual", "Safety"]),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config()).chunk(_doc(els))
    assert len(chunks) == 1
    assert chunks[0].provenance.header_path == ["Manual", "Safety"]  # h1 not lost


def test_geometry_boxes_survive_into_the_chunk():
    els = [
        _el("h", ElementKind.HEADING, "S", level=1),
        _el("p1", ElementKind.PARAGRAPH, "Wear PPE.", header_path=["S"], boxes=[(1, (10, 20, 200, 35))]),
        _el("p2", ElementKind.PARAGRAPH, "Lock out power.", header_path=["S"], boxes=[(1, (10, 40, 200, 55))]),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config(max_chars=20)).chunk(_doc(els))
    leaf = chunks[1]  # first leaf
    assert leaf.provenance.geometry[0].boxes[0].page == 1  # the PDF page box rode through the merge


def test_oversize_element_is_character_split():
    big = "word " * 100  # 500 chars
    chunks = StructureAwareChunker(StructureAwareChunker.Config(max_chars=100)).chunk(
        _doc([_el("p", ElementKind.PARAGRAPH, big, start=0, end=len(big))], text=big)
    )
    assert len(chunks) > 1
    assert all(c.provenance.source_element_ids == ["p"] for c in chunks)  # all map back to the one element


def test_nested_subsections_build_a_multi_level_tree():
    # Manual (h1) ⊃ Safety (h2: an intro + the PPE subsection) ⊃ PPE (h3: two leaves).
    els = [
        _el("manual", ElementKind.HEADING, "Manual", level=1),
        _el("safety", ElementKind.HEADING, "Safety", header_path=["Manual"], level=2),
        _el("intro", ElementKind.PARAGRAPH, "Be careful.", header_path=["Manual", "Safety"]),
        _el("ppe", ElementKind.HEADING, "PPE", header_path=["Manual", "Safety"], level=3),
        _el("p1", ElementKind.PARAGRAPH, "Wear gloves.", header_path=["Manual", "Safety", "PPE"]),
        _el("p2", ElementKind.PARAGRAPH, "Use goggles.", header_path=["Manual", "Safety", "PPE"]),
    ]
    chunks = StructureAwareChunker(StructureAwareChunker.Config(max_chars=20)).chunk(_doc(els))

    assert {c.provenance.level for c in chunks} == {0, 1, 2}  # leaf -> PPE parent (1) -> Safety parent (2)

    safety = next(c for c in chunks if c.provenance.level == 2)
    assert safety.provenance.header_path == ["Manual", "Safety"] and safety.parent_index is None  # tree top
    ppe = next(c for c in chunks if c.provenance.level == 1)
    assert ppe.provenance.header_path == ["Manual", "Safety", "PPE"]
    assert chunks[ppe.parent_index] is safety  # the PPE subsection merges up into Safety

    leaf = next(c for c in chunks if c.provenance.source_element_ids == ["p1"])
    assert chunks[leaf.parent_index] is ppe  # a leaf merges up into PPE, then Safety

    assert all(c.provenance.header_path != ["Manual"] for c in chunks)  # the single-child h1 collapsed away


# --- recursive fallback ------------------------------------------------------- #

def test_recursive_chunker_threads_provenance():
    text = "First section body. " * 10
    els = [_el("p", ElementKind.PARAGRAPH, text, header_path=["Body"], start=0, end=len(text))]
    chunks = RecursiveCharacterChunker(RecursiveCharacterChunker.Config(chunk_size=40, overlap=8)).chunk(
        _doc(els, text=text)
    )
    assert len(chunks) > 1
    assert all(c.provenance.level == 0 and c.parent_index is None for c in chunks)  # flat
    assert all(c.provenance.content_hash for c in chunks)
    assert any(c.provenance.source_element_ids == ["p"] for c in chunks)  # mapped to the overlapping element
    assert any(c.provenance.header_path == ["Body"] for c in chunks)


# --- the ChunkStage driver ---------------------------------------------------- #

def test_driver_defaults_to_structure_aware():
    stage = ChunkStage(ChunkStage.Config())
    list(stage.process(PipelineItem(content="hi", metadata={"doc_id": "d1"})))
    assert isinstance(stage._chunker, StructureAwareChunker)


def test_driver_synthesizes_a_document_when_none_present():
    stage = ChunkStage(ChunkStage.Config())
    out = list(stage.process(PipelineItem(content="A plain sentence with no extraction.", metadata={"doc_id": "d1"})))
    assert len(out) == 1
    item = out[0]
    assert item.provenance is not None and item.provenance.content_hash
    assert item.metadata["chunk_index"] == 0 and item.metadata["total_chunks"] == 1
    assert item.metadata["chunk_level"] == 0 and item.metadata["parent_ordinal"] is None


def test_driver_emits_tree_metadata_for_persistence():
    els = [
        _el("h", ElementKind.HEADING, "Safety", level=1),
        _el("p1", ElementKind.PARAGRAPH, "Wear PPE.", header_path=["Safety"]),
        _el("p2", ElementKind.PARAGRAPH, "Lock out power.", header_path=["Safety"]),
    ]
    stage = ChunkStage(ChunkStage.Config(chunker={"class_name": "structure_aware", "max_chars": 20}))
    out = list(stage.process(PipelineItem(content="", metadata={"doc_id": "d1"}, document=_doc(els))))
    parent, leaf1, leaf2 = out
    assert parent.metadata["chunk_level"] == 1 and parent.metadata["parent_ordinal"] is None
    assert leaf1.metadata["parent_ordinal"] == 0 and leaf2.metadata["parent_ordinal"] == 0  # point at the parent's ordinal


def test_every_parent_geometry_contains_its_descendants_spans_verbatim():
    """The merger detects "a returned section already covers this result" by a subset test on persisted
    ``(start, end)`` spans (``AutoMerger._drop_covered``), so each section parent's geometry must hold
    every descendant's spans verbatim — if the chunker ever coalesced adjacent spans, the merger would
    silently stop recognizing a section's own fragments. Pinned here, in the module that owns it."""
    from tarnrag.ingestion.components.extraction.extractor import Source
    from tarnrag.ingestion.components.extraction.markdown import MarkdownExtractor

    md = (
        "# Pumps\n\nInspect pump seals monthly.\n\nReplace pump bearings yearly.\n\n"
        "## Valves\n\nCheck valve torque weekly.\n\nLubricate valve stems often.\n\n"
        "### Gaskets\n\nSwap the flange gaskets.\n\nTorque the flange bolts.\n"
    )
    doc = MarkdownExtractor(MarkdownExtractor.Config()).extract(Source(source_id="d", source_kind="md", content=md))
    chunks = StructureAwareChunker(StructureAwareChunker.Config(max_chars=40)).chunk(doc)
    assert max(c.provenance.level for c in chunks) >= 2  # a real multi-level tree

    def spans(chunk):
        return {(s.start, s.end) for s in chunk.provenance.geometry}

    for child in chunks:
        ancestor = child.parent_index
        while ancestor is not None:  # every ancestor, not just the direct parent
            assert spans(child) <= spans(chunks[ancestor]), (child.text, chunks[ancestor].text)
            ancestor = chunks[ancestor].parent_index
