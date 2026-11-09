"""CRIT-02: CleanAndNormalize cleans the STRUCTURED document — the substrate the chunker splits and every
span indexes into — not just ``item.content`` (which nothing downstream read), and the persisted document
is the cleaned one, so chunk spans index ``documents.content``."""

from tarnrag.contracts import (
    Annotation,
    Element,
    ElementKind,
    PipelineItem,
    Span,
    StructuredDocument,
)
from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.components.extraction.extractor import Source
from tarnrag.ingestion.components.extraction.markdown import MarkdownExtractor
from tarnrag.ingestion.engine.result_sink import DocumentRefreshSink
from tarnrag.ingestion.pipeline.clean_normalize import (
    CleanAndNormalizeStage,
    clean_text,
)


def _stage(**config) -> CleanAndNormalizeStage:
    return CleanAndNormalizeStage(CleanAndNormalizeStage.Config(**config))


def _clean(document: StructuredDocument) -> StructuredDocument:
    [out] = _stage().process(PipelineItem(content=document.text, metadata={}, document=document))
    assert out.content == out.document.text and out.metadata["cleaned"] is True
    return out.document


def _assert_spans_index_the_text(doc: StructuredDocument) -> None:
    for el in doc.elements:
        [span] = el.geometry
        assert doc.text[span.start : span.end] == el.text, el.id


def test_the_string_rules():
    assert clean_text("  Hello    world\r\n\r\n\r\nFoo\t\tbar\x00\x07  ") == "Hello world\n\nFoo bar"
    assert clean_text("a  b\n\n\n\nc", collapse_whitespace=False) == "a  b\n\n\n\nc"  # only controls/edges


def test_markdown_document_is_cleaned_with_spans_kept_exact_and_code_untouched():
    md = "# Title\n\nSome    spaced\ttext.\n\n```\ndef f():\n        return  1\n```\n\nAfter   code."
    source = MarkdownExtractor(MarkdownExtractor.Config()).extract(
        Source(source_id="d", source_kind="md", content=md)
    )
    doc = _clean(source)
    texts = {el.kind: el.text for el in doc.elements if el.kind != ElementKind.PARAGRAPH}
    paragraphs = [el.text for el in doc.elements if el.kind == ElementKind.PARAGRAPH]
    assert paragraphs == ["Some spaced text.", "After code."]
    assert texts[ElementKind.CODE] == "def f():\n        return  1"  # indentation is meaning in code
    _assert_spans_index_the_text(doc)
    assert "Some    spaced" in source.text  # the upstream document was not mutated


def test_annotation_and_sentence_spans_follow_the_text():
    text = "Wear     PPE  (Personal Protective Equipment)."
    start = text.index("PPE")
    el = Element(
        id="e0", kind=ElementKind.PARAGRAPH, text=text, geometry=[Span(start=0, end=len(text))],
        sentences=[Span(start=0, end=len(text))],
        annotations=[Annotation(producer="acronyms", type="acronym", span=[Span(start=start, end=start + 3)])],
    )
    doc = _clean(StructuredDocument(
        source_id="d", source_kind="text", extractor="t", text=text, elements=[el], content_hash="h",
        annotations=[Annotation(producer="doc", type="topic", span=[Span(start=start, end=start + 3)])],
    ))
    [cleaned] = doc.elements
    [ann] = cleaned.annotations[0].span
    assert doc.text == "Wear PPE (Personal Protective Equipment)."
    assert doc.text[ann.start : ann.end] == "PPE"
    [doc_ann] = doc.annotations[0].span
    assert doc.text[doc_ann.start : doc_ann.end] == "PPE"
    assert cleaned.sentences[0].end == len(doc.text)


def test_an_item_without_a_document_cleans_just_its_text():
    [out] = _stage().process(PipelineItem(content="a   b\x00", metadata={"k": 1}))
    assert out.content == "a b" and out.document is None and out.metadata == {"k": 1, "cleaned": True}


async def test_the_refresh_sink_never_writes_a_chunk_phase_item_as_a_document():
    class _Store:
        def __init__(self) -> None:
            self.stored: list = []

        async def store_documents(self, docs):
            self.stored.extend(docs)
            return [d.metadata["source_id"] for d in docs]

    store = _Store()
    sink = DocumentRefreshSink(store)
    chunk_item = PipelineItem(content="chunk", metadata={"source_id": "d"})  # no document: chunk phase
    sink.submit([chunk_item])
    sink.close()
    assert (await sink.finalize()).persisted and store.stored == []


async def test_default_ingest_chunks_and_persists_the_cleaned_text(tmp_path):
    """The review's acceptance case, end to end through the default pipeline (hash embedder, no model)."""
    settings = Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=32,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/clean.db"),
    )
    async with await IngestionEngine.create(settings) as engine:
        await engine.ingest_content([{"source_id": "d", "content": "Hello    world\r\n\r\n\r\nFoo\t\tbar"}])
        stored = await engine.repository.get_document("d")
        [chunk] = await engine.repository.get_chunks_by_document("d")
        assert (await engine.status("d")).status == "complete"
    assert stored.content == "Hello world\n\nFoo bar"
    assert chunk.content == "Hello world\n\nFoo bar"
    [span] = chunk.provenance.geometry
    assert stored.content[span.start : span.end] == chunk.content  # spans index the persisted text


def test_a_table_keeps_its_grid_rendering():
    """A table's text is the extractor's rendering of its cells — whitespace there is layout, not noise."""
    from tarnrag.ingestion.components.extraction.table_json import TableJsonExtractor

    grid = '[["", "2019", "2018"], ["Revenue", "$10", "$8"]]'
    source = TableJsonExtractor(TableJsonExtractor.Config()).extract(
        Source(source_id="t", source_kind="table", content=grid)
    )
    doc = _clean(source)
    assert doc.text == source.text == " | 2019 | 2018\nRevenue | $10 | $8"  # leading empty cell kept
    assert doc.elements[0].table.markdown == doc.text
    _assert_spans_index_the_text(doc)
