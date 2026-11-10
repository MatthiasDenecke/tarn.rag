import pytest
from bausatz import ComponentFactory
from pydantic import ValidationError

from tarnrag.contracts import ChunkProvenance, PipelineItem
from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.ingestion.components.chunking.chunk import ChunkStage
from tarnrag.ingestion.components.enrichment.enrich import EnrichStage
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.embed import EmbedStage
from tarnrag.ingestion.pipeline.pipeline import Pipeline


def _item(content, **meta):
    return PipelineItem(content=content, metadata=meta)


def test_load_from_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello from a file")
    out = list(LoadAndParseStage(LoadAndParseStage.Config()).process(_item("", source_path=str(p))))
    assert len(out) == 1
    assert out[0].content == "hello from a file"
    assert out[0].metadata["doc_id"] and out[0].metadata["loaded"] is True


def test_load_passthrough_content():
    out = list(LoadAndParseStage(LoadAndParseStage.Config()).process(_item("already loaded")))
    assert out[0].content == "already loaded"


def test_clean_normalizes_whitespace_and_controls():
    stage = CleanAndNormalizeStage(CleanAndNormalizeStage.Config())
    out = list(stage.process(_item("  a\x07b   c\t\td  ")))
    assert out[0].content == "ab c d"
    assert out[0].metadata["cleaned"] is True


def _recursive(size, overlap):
    """A Chunk stage driving the recursive chunker (the structure-agnostic, size-based strategy)."""
    return ChunkStage(ChunkStage.Config(chunker={"class_name": "recursive", "chunk_size": size, "overlap": overlap}))


def test_chunk_splits_and_indexes():
    stage = _recursive(20, 4)
    text = "abcdefghij " * 8  # ~88 chars -> several chunks
    out = list(stage.process(_item(text, doc_id="d1")))
    assert len(out) > 1
    total = out[0].metadata["total_chunks"]
    assert total == len(out)
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all(o.provenance is not None and o.provenance.content_hash for o in out)  # each chunk carries provenance


def test_chunk_validation():
    # The size/overlap validation lives on the recursive chunker's Config (pydantic).
    from tarnrag.ingestion.components.chunking.recursive import (
        RecursiveCharacterChunker,
    )

    with pytest.raises(ValidationError):
        RecursiveCharacterChunker.Config(chunk_size=0)
    with pytest.raises(ValidationError):
        RecursiveCharacterChunker.Config(chunk_size=10, overlap=10)


class _FakeEmbedder:
    """Deterministic 3-d embedder that records its batch calls."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_embed_produces_embeddings_with_model_batching():
    stage = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(batch_size=2)))
    stage._embedder = _FakeEmbedder()
    items = [
        _item(f"chunk {i}", chunk_id=f"c{i}", source_id="s1") for i in range(3)
    ]
    embs = list(stage.process_batch(items))
    assert [e.chunk_id for e in embs] == ["c0", "c1", "c2"]
    assert all(len(e.vector) == 3 for e in embs)  # just the key + vector — identity is index-wide
    # batch_size=2 over 3 items -> two embed calls (two-tier batching)
    assert len(stage._embedder.calls) == 2


def test_embed_injects_header_path_when_enabled():
    """inject_header_path prepends the chunk's section header path to its text before embedding;
    queries are never injected and a chunk with no header path is embedded as-is."""
    item = PipelineItem(
        content="Wear gloves.", metadata={"chunk_id": "c0", "source_id": "s1"},
        provenance=ChunkProvenance(content_hash="h", header_path=["Safety", "PPE"]),
    )
    on = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(inject_header_path=True)))
    on._embedder = _FakeEmbedder()
    list(on.process_batch([item]))
    assert on._embedder.calls[-1] == ["Safety > PPE\nWear gloves."]  # header path prepended

    off = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(inject_header_path=False)))
    off._embedder = _FakeEmbedder()
    list(off.process_batch([item]))
    assert off._embedder.calls[-1] == ["Wear gloves."]  # default: content only

    # Injection on, but a plain-text item with no header path -> content embedded unchanged.
    on2 = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(inject_header_path=True)))
    on2._embedder = _FakeEmbedder()
    list(on2.process_batch([_item("plain text", chunk_id="c1", source_id="s1")]))
    assert on2._embedder.calls[-1] == ["plain text"]


def test_embed_contextualizes_table_chunks_when_enabled():
    """contextualize_tables embeds a TABLE chunk as its header-contextualized rendering (values bound
    to headers) instead of the raw grid; stored text is untouched, non-table chunks are unaffected,
    and header-path injection composes on top."""
    from tarnrag.contracts import Table, TableCell

    table = Table(n_rows=2, n_cols=2, cells=[
        TableCell(id="h", row=0, col=1, is_column_header=True, text="2019"),
        TableCell(id="r", row=1, col=0, is_row_header=True, text="Goodwill"),
        TableCell(id="v", row=1, col=1, text="1,910"),
    ])
    item = PipelineItem(
        content="| 2019 |\n| Goodwill | 1,910 |", metadata={"chunk_id": "c0", "source_id": "s1"},
        provenance=ChunkProvenance(content_hash="h", header_path=["Notes"], table=table),
    )
    on = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings(contextualize_tables=True)))
    on._embedder = _FakeEmbedder()
    list(on.process_batch([item]))
    assert on._embedder.calls[-1] == ["Goodwill \u2014 2019: 1,910"]  # rendered from cells, not the grid

    off = EmbedStage(EmbedStage.Config(embedding=EmbeddingSettings()))
    off._embedder = _FakeEmbedder()
    list(off.process_batch([item]))
    assert off._embedder.calls[-1] == [item.content]  # default: the grid text as stored

    both = EmbedStage(EmbedStage.Config(
        embedding=EmbeddingSettings(contextualize_tables=True, inject_header_path=True)
    ))
    both._embedder = _FakeEmbedder()
    list(both.process_batch([item]))
    assert both._embedder.calls[-1] == ["Notes\nGoodwill \u2014 2019: 1,910"]  # the two compose


def test_stage_built_from_dict_spec_via_component_factory():
    """The migration's goal: a stage (and its chunker child) instantiated from a plain dict spec."""
    chunk = ComponentFactory.get().create(
        {"class_name": "Chunk", "chunker": {"class_name": "recursive", "chunk_size": 16, "overlap": 4}}
    )
    assert isinstance(chunk, ChunkStage)
    assert (chunk._chunker.config.chunk_size, chunk._chunker.config.overlap) == (16, 4)  # child built by the factory
    assert chunk.tag == "Chunk"  # the type tag (the sink/metrics key)
    assert chunk.name == "Chunk"  # unnamed ⇒ the bare tag; PipelineDAG adds the positional suffix
    assert chunk.to_dict()["class_name"] == "Chunk"  # round-trips back to a spec


def test_pipeline_composition_doc_to_chunks():
    pipe = Pipeline.from_stages(
        [
            LoadAndParseStage(LoadAndParseStage.Config()),
            EnrichStage(EnrichStage.Config(enrichers=[{"class_name": "acronyms"}])),  # doc-phase
            CleanAndNormalizeStage(CleanAndNormalizeStage.Config()),
            _recursive(20, 4),
        ]
    )
    item = _item("Always wear PPE. " * 6, source_id="s1")
    out = list(pipe.run([item]))
    assert len(out) >= 1
    total = out[0].metadata["total_chunks"]
    assert [o.metadata["chunk_index"] for o in out] == list(range(total))
    assert all("doc_id" in o.metadata for o in out)
    # the enricher's annotation rode through to the chunks' provenance (extract -> enrich -> chunk)
    assert any(a.value.get("text") == "PPE" for o in out for a in o.provenance.annotations)
