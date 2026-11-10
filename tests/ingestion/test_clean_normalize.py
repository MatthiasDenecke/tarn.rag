"""CRIT-02: CleanAndNormalize cleans the STRUCTURED document — the substrate the chunker splits and every
span indexes into — not just ``item.content`` (which nothing downstream read), and the persisted document
is the cleaned one, so chunk spans index ``documents.content``."""

import random

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
    _Protected,
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


def test_an_element_with_two_spans_keeps_exact_spans_and_cleans_its_text_on_its_own():
    """No built-in extractor emits one, but a custom extractor may (a paragraph broken by a page header).
    Pins the non-exact path: each span indexes the cleaned text exactly, while the element's text is cleaned
    standalone — so it can differ from the slices at an outer edge the document keeps."""
    first, second = "Lockout\t\ttagout  procedure ", "applies   to\x00 every valve "
    text = f"{first}\n\n\n\nPage 2 header\n\n\n\n{second}\nFooter"
    at = text.index(second)
    el = Element(
        id="e0", kind=ElementKind.PARAGRAPH, text=first + second,
        geometry=[Span(start=0, end=len(first)), Span(start=at, end=at + len(second))],
    )
    doc = _clean(StructuredDocument(
        source_id="d", source_kind="text", extractor="t", text=text, elements=[el], content_hash="h",
    ))
    [cleaned] = doc.elements
    slices = [doc.text[s.start : s.end] for s in cleaned.geometry]
    assert slices == ["Lockout tagout procedure ", "applies to every valve "]
    assert cleaned.text == "Lockout tagout procedure applies to every valve"  # cleaned on its own
    assert "".join(slices) == cleaned.text + " "  # the pinned difference: its outer edge was stripped


def test_the_protected_range_lookup_matches_the_linear_scan():
    """The bisect over sorted, merged ranges answers exactly what checking every range did."""
    rng = random.Random(7)
    for _ in range(300):
        starts = [rng.randint(0, 100) for _ in range(rng.randint(0, 12))]
        ranges = [(s, s + rng.randint(0, 8)) for s in starts]  # unsorted, overlapping, some empty
        guard = _Protected(ranges)
        for _ in range(30):
            start = rng.randint(0, 110)
            end = start + rng.randint(0, 10)
            assert guard.overlaps(start, end) == any(s < end and start < e for s, e in ranges)


async def test_the_refresh_sink_never_writes_a_chunk_phase_item_as_a_document():
    class _Store:
        def __init__(self) -> None:
            self.stored: list = []

        async def refresh_documents(self, docs):
            self.stored.extend(docs)
            return [d.metadata["source_id"] for d in docs]

    store = _Store()
    sink = DocumentRefreshSink(store)
    chunk_item = PipelineItem(content="chunk", metadata={"source_id": "d"})  # no document: chunk phase
    sink.submit([chunk_item])
    sink.close()
    assert (await sink.finalize()).persisted and store.stored == []


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=32,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/clean.db"),
    )


async def test_a_replayed_clean_job_keeps_the_chunks_and_vectors_built_after_it(tmp_path):
    """At-least-once redelivery can replay a Clean job after its Chunk/Embed jobs finished. Its refresh
    rewrites the document row only — it must not cascade-delete the chunks and embeddings."""
    md = "# Pumps\n\nInspect   the seals.\n\n## Valves\n\nCheck\tvalve torque.\n"
    async with await IngestionEngine.create(_settings(tmp_path)) as engine:
        await engine.ingest_content([{"source_id": "d", "content": md, "source_type": "md"}])
        repo = engine.repository
        before = await repo.document_facts("d")
        stored = await repo.get_document("d")
        replayed = PipelineItem(  # what the redelivered job's stage re-emits: the same cleaned document
            content=stored.content,
            metadata={**stored.metadata, "source_id": "d"},
            document=StructuredDocument(
                source_id="d", source_kind="md", extractor="markdown", text=stored.content, content_hash="h"
            ),
        )
        sink = DocumentRefreshSink(repo)
        sink.submit([replayed])
        sink.close()
        assert (await sink.finalize()).persisted
        assert await repo.document_facts("d") == before  # every chunk and vector survived
        assert before.embedding_count and (await engine.status("d")).status == "complete"
        assert (await repo.get_document("d")).content == stored.content


async def test_default_ingest_chunks_and_persists_the_cleaned_text(tmp_path):
    """The review's acceptance case, end to end through the default pipeline (hash embedder, no model)."""
    settings = _settings(tmp_path)
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
