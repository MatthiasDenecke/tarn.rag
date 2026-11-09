"""AutoMerger over the multi-level section tree — permitted-chunk filtering of the parents it hydrates
(MED-01), iterative merging up the tree, and dropping a result a returned ancestor already covers.

Unit-level: a fake store serves the section parents; the results stand in for retrieved leaves."""

from tarnrag.contracts import ChunkProvenance, ChunkRecord, MethodRef, RetrievalResult, Span
from tarnrag.retrieval import AutoMerger, Query, RetrievalContext
from tarnrag.retrieval.types import Purpose


# Each named piece of text sits at its own char span; a chunk's geometry is the spans of what it covers
# (a section parent's = its heading + every descendant's — the chunker's aggregation, persisted).
_AT = {name: (i * 10, i * 10 + 5) for i, name in enumerate("abcdefgh")} | {
    name: (100 + i * 10, 105 + i * 10) for i, name in enumerate(["h", "h1", "h2", "hg", "e7"])
}


def _prov(cid: str, *, level: int = 0, parent: str | None = None, elements: list[str] | None = None):
    return ChunkProvenance(
        content_hash=cid, level=level, parent_chunk_id=parent,
        geometry=[Span(start=_AT[e][0], end=_AT[e][1]) for e in (elements or [cid])],
    )


def _leaf(cid: str, score: float, *, parent: str, elements: list[str] | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid, text=cid, score=score, component_scores={"dense": score},
        document_id="d", source_kind="document", standard_id=None, locator=None,
        license_class="public_domain", methods=[MethodRef("m", "1")],
        provenance=_prov(cid, parent=parent, elements=elements),
    )


def _section(
    cid: str, *, level: int, elements: list[str], parent: str | None = None, **record: object
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid, text=f"section {cid}", document_id="d", source_kind="document", standard_id=None,
        locator=None, license_class=str(record.pop("license_class", "public_domain")),
        provenance=_prov(cid, level=level, parent=parent, elements=elements), **record,
    )


class _Store:
    """Hydrates the given section records by id (and counts the round trips)."""

    def __init__(self, *records: ChunkRecord) -> None:
        self._by_id = {r.chunk_id: r for r in records}
        self.calls = 0

    async def hydrate(self, ids: list[str]) -> list[ChunkRecord]:
        self.calls += 1
        return [self._by_id[i] for i in ids if i in self._by_id]


def _ctx(store: _Store) -> RetrievalContext:
    return RetrievalContext(store=store, embedder=None)  # no license policy ⇒ Query.permitted_filter


async def _merge(store: _Store, results: list[RetrievalResult], query: Query | None = None) -> list[str]:
    out = await AutoMerger(AutoMerger.Config()).merge(query or Query(text="q"), results, _ctx(store))
    return [r.chunk_id for r in out]


async def test_siblings_merge_into_a_permitted_parent():
    store = _Store(_section("p", level=1, elements=["h", "a", "b"], methods=[("m", "1")]))
    assert await _merge(store, [_leaf("a", 0.9, parent="p"), _leaf("b", 0.8, parent="p")]) == ["p"]


async def test_a_parent_outside_the_method_scope_is_not_substituted():
    """A method bundle may hold the leaves but not their section: substituting the parent would return a
    chunk the scoped query may not see (and the siblings' text it carries)."""
    store = _Store(_section("p", level=1, elements=["h", "a", "b"], methods=[]))
    scoped = Query(text="q", scope=[MethodRef("m")])
    leaves = [_leaf("a", 0.9, parent="p"), _leaf("b", 0.8, parent="p")]
    assert await _merge(store, leaves, scoped) == ["a", "b"]
    assert await _merge(store, leaves) == ["p"]  # unscoped, the same parent is fine


async def test_a_grounding_disallowed_parent_is_not_substituted_for_generation():
    store = _Store(_section("p", level=1, elements=["h", "a", "b"], ai_grounding_allowed=False))
    leaves = [_leaf("a", 0.9, parent="p"), _leaf("b", 0.8, parent="p")]
    grounding = Query(text="q", purpose=Purpose.GENERATION_GROUNDING)
    assert await _merge(store, leaves, grounding) == ["a", "b"]
    assert await _merge(store, leaves) == ["p"]  # EXECUTION may see it


async def test_merging_climbs_the_tree_and_drops_what_an_ancestor_covers():
    """g ⊃ {p1 ⊃ {a, b}, c, p2 ⊃ {d, e}}. Retrieved: a, b, c, d. Pass 1 merges a+b → p1; pass 2 merges
    p1+c → g (they share g). d's own parent p2 never qualified, but g covers d's text — so d drops."""
    g_elements = ["hg", "h1", "a", "b", "c", "h2", "d", "e"]
    store = _Store(
        _section("p1", level=1, elements=["h1", "a", "b"], parent="g"),
        _section("p2", level=1, elements=["h2", "d", "e"], parent="g"),
        _section("g", level=2, elements=g_elements),
    )
    results = [
        _leaf("a", 0.9, parent="p1"), _leaf("d", 0.85, parent="p2"),
        _leaf("b", 0.8, parent="p1"), _leaf("c", 0.7, parent="g"),
    ]
    out = await AutoMerger(AutoMerger.Config()).merge(Query(text="q"), results, _ctx(store))
    assert [r.chunk_id for r in out] == ["g"]
    assert out[0].score == 0.9  # the best descendant's score carries up
    assert store.calls == 2  # one hydrate per productive pass


async def test_a_refused_parent_is_not_retried_and_the_loop_ends():
    store = _Store()  # the parent is gone
    leaves = [_leaf("a", 0.9, parent="p"), _leaf("b", 0.8, parent="p")]
    assert await _merge(store, leaves) == ["a", "b"]
    assert store.calls == 1


async def test_a_section_retrieved_with_its_own_fragment_keeps_only_the_section():
    """From an index that still holds section vectors, a section and one of its leaves can both be
    retrieved (below the merge threshold) — the leaf is redundant next to the section that contains it."""
    section = RetrievalResult.from_record(
        _section("p", level=1, elements=["h", "a", "b"]), score=0.95, component_scores={"dense": 0.95}
    )
    out = await AutoMerger(AutoMerger.Config()).merge(
        Query(text="q"), [section, _leaf("a", 0.9, parent="p")], _ctx(_Store())
    )
    assert [r.chunk_id for r in out] == ["p"]


async def test_oversize_pieces_of_one_element_are_not_mistaken_for_containment():
    """Two pieces of one character-split element share its id at the same level — neither covers the other."""
    first = _leaf("a#0", 0.9, parent="p", elements=["e7"])
    second = _leaf("a#1", 0.8, parent="q", elements=["e7"])
    assert await _merge(_Store(), [first, second]) == ["a#0", "a#1"]


async def test_multi_level_merge_over_persisted_chunks(tmp_path):
    """The same climb end to end, over what ingestion actually persists and hydration actually returns
    (geometry survives hydration; source element ids don't). Pumps ⊃ {two leaves, Valves ⊃ {two leaves}}:
    retrieving all four leaves merges each section and then drops Valves, which Pumps already carries."""
    from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
    from tarnrag.ingestion import IngestionEngine
    from tarnrag.retrieval import RetrievalPipeline

    spec = {"class_name": "pipeline", "stages": [
        {"class_name": "LoadAndParse"},
        {"class_name": "Chunk", "chunker": {"class_name": "structure_aware", "max_chars": 40}},
        {"class_name": "Embed"},
    ]}
    settings = Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=16,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/merge.db"),
        components={"ingestion_pipeline": spec},
    )
    md = (
        "# Pumps\n\nInspect pump seals monthly.\n\nReplace pump bearings yearly.\n\n"
        "## Valves\n\nCheck valve torque weekly.\n\nLubricate valve stems often.\n"
    )
    async with await IngestionEngine.create(settings) as ingest:
        await ingest.ingest_content([{"source_id": "m", "content": md, "source_type": "md"}])
        chunks = await ingest.repository.get_chunks_by_document("m")
        top = next(c for c in chunks if c.provenance.level == 2)
        pipeline = RetrievalPipeline(RetrievalPipeline.Config(
            retrievers=[{"class_name": "sparse"}], merger={"class_name": "auto_merge"},
        ))
        ctx = RetrievalContext(store=ingest.repository, embedder=None)
        results = await pipeline.search(Query(text="pump valve seals bearings torque stems"), ctx)
    assert [r.chunk_id for r in results] == [top.id]
    assert results[0].provenance.header_path == ["Pumps"]
