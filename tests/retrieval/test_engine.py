"""RetrievalEngine (dense-only) over the §8 repository, built with a fake embedder."""

import pytest

from tarnrag.contracts import IndexMeta
from tarnrag.contracts import (
    Candidate, Chunk, ChunkFilter, ChunkProvenance, ChunkRecord, Document, Embedding, MethodRef,
    RetrievalResult,
)
from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, Settings
from tarnrag.retrieval.components.fuser import IdentityFuser, RRFFuser
from tarnrag.retrieval import (
    AutoMerger, CrossEncoderReranker, DefaultLicensePolicy, Query, RetrievalContext, RetrievalEngine,
    RetrievalError, RetrievalPipeline,
)
from tarnrag.retrieval.types import Purpose, SearchTrace

FINGERPRINT = "fp-123"


class _FakeEmbedder:
    """Embeds a query to a fixed 3-d vector; fingerprint matches the index by default."""

    def __init__(self, query_vec=(1.0, 0.0, 0.0), fingerprint=FINGERPRINT):
        self._vec = list(query_vec)
        self._fp = fingerprint

    def embed_query(self, text):
        return self._vec

    def config_fingerprint(self):
        return self._fp

    def embed_meta(self):
        return {"embedding_dim": "3", "embedding_config_fingerprint": FINGERPRINT}


class _FakeCrossEncoder:
    """Scores passages by a text→score lookup (query ignored); higher = more relevant."""

    def __init__(self, by_text):
        self._by_text = by_text

    def score(self, query, passages):
        return [self._by_text[p] for p in passages]

    def identity(self):
        return "fake-ce"


async def _index(repo):
    """Stamp index_meta and ingest two chunks (+ embeddings) into the repo; return the chunk ids."""
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))
    _, cids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1", "title": "T"}),
        [
            Chunk(parent_doc_id="s1", content="storage tank inspection",
                  chunk_index=0, total_chunks=2, metadata={"locator": "§6.4"}),
            Chunk(parent_doc_id="s1", content="quokka marsupial",
                  chunk_index=1, total_chunks=2, metadata={}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0]),
    ])
    return cids


async def test_open_validates_fingerprint(repo):
    await _index(repo)
    # Matching fingerprint -> opens.
    await RetrievalEngine.open(repo, _FakeEmbedder())
    # Mismatched fingerprint -> refuses.
    with pytest.raises(RetrievalError, match="fingerprint mismatch"):
        await RetrievalEngine.open(repo, _FakeEmbedder(fingerprint="other"))


async def test_open_with_settings_wires_an_llm_for_the_bridge(repo):
    await _index(repo)
    # Built from Settings -> a (lazy, unused-unless-a-bridge-component-calls-it) LLM rides the context.
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(), Settings(_env_file=None, EMBEDDING_DIMENSION=3))
    assert engine._llm is not None
    # Settings-less open (the low-level seam) wires no LLM — the lean default path needs none.
    assert (await RetrievalEngine.open(repo, _FakeEmbedder()))._llm is None


async def test_search_ranks_nearest_with_provenance(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))

    results = await engine.search(Query(text="tank inspection", top_k=2))
    assert [r.chunk_id for r in results] == cids  # chunk 0 ([1,0,0]) nearest
    top = results[0]
    assert top.text == "storage tank inspection"
    assert top.document_id == "s1"
    assert top.source_kind == "document"
    assert top.license_class == "public_domain"
    assert top.locator == "§6.4"
    assert top.methods == []  # method_chunks empty for now
    assert top.score >= results[1].score  # ranked by score desc
    assert "dense" in top.component_scores


async def test_top_k_truncates(repo):
    await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert len(await engine.search(Query(text="x", top_k=1))) == 1


async def test_empty_index_returns_empty(repo):
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))  # built, but no chunks
    engine = await RetrievalEngine.open(repo, _FakeEmbedder())
    assert await engine.search(Query(text="x")) == []


async def test_search_text_convenience(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = await engine.search_text("tank inspection", top_k=2)
    assert [r.chunk_id for r in results] == cids


async def test_open_refuses_unbuilt_index(repo):
    # No write_index_meta -> the index has not been built yet.
    with pytest.raises(RetrievalError, match="not been built"):
        await RetrievalEngine.open(repo, _FakeEmbedder())


async def test_pipeline_hybrid_fuses_dense_and_sparse(repo):
    cids = await _index(repo)  # cids[0] = "storage tank inspection", cids[1] = "quokka marsupial"
    pipe = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}, {"class_name": "sparse"}], fuser={"class_name": "rrf"}
        )
    )
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    results = await pipe.search(Query(text="tank inspection", top_k=2), ctx)
    assert results[0].chunk_id == cids[0]  # the tank chunk: top dense AND top sparse
    assert set(results[0].component_scores) == {"dense", "sparse"}  # both retrievers contributed
    assert results[0].provenance is not None  # provenance threaded through the read path


def test_retriever_keys_disambiguate_duplicates():
    """Two same-class retrievers must map to distinct keys, so neither's candidates are silently dropped
    when building the per-retriever map; a configured `name` is honoured."""
    pipe = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}, {"class_name": "dense"}, {"class_name": "sparse"}]
        )
    )
    pipe.ensure_built()
    assert pipe._retriever_keys() == ["dense", "dense#1", "sparse"]  # duplicate disambiguated by suffix
    named = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense", "name": "primary"}, {"class_name": "dense", "name": "backup"}]
        )
    )
    named.ensure_built()
    assert named._retriever_keys() == ["primary", "backup"]  # explicit names used as-is


async def test_engine_uses_configured_pipeline_spec(repo):
    cids = await _index(repo)
    settings = Settings(
        _env_file=None,
        components={
            RETRIEVAL_PIPELINE: {
                "class_name": "retrieval_pipeline",
                "retrievers": [{"class_name": "sparse"}],
                "fuser": {"class_name": "identity"},
            }
        },
    )
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(), settings=settings)
    results = await engine.search(Query(text="tank inspection", top_k=5))
    assert [r.chunk_id for r in results] == [cids[0]]  # sparse-only -> just the lexical match
    assert set(results[0].component_scores) == {"sparse"}


async def test_filter_drops_unavailable_and_respects_grounding(repo):
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="tank one", chunk_index=0, total_chunks=2,
                  metadata={"available": 0}),  # unavailable
            Chunk(parent_doc_id="s1", content="tank two", chunk_index=1, total_chunks=2,
                  metadata={"ai_grounding_allowed": 0}),  # not grounding-allowed
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=a, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=b, vector=[0.9, 0.1, 0.0]),
    ])
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    # EXECUTION: the unavailable chunk is dropped; the grounding-disallowed one is kept.
    assert [r.chunk_id for r in await engine.search(Query(text="tank", top_k=5))] == [b]
    # GENERATION_GROUNDING: a is unavailable AND b isn't grounding-allowed -> nothing survives.
    assert await engine.search(Query(text="tank", top_k=5, purpose=Purpose.GENERATION_GROUNDING)) == []


def test_query_permitted_filter_maps_purpose_and_scope():
    """The Query builds the permitted-chunk filter the retrievers pass to the store (the filtering moved
    into the retrievers): available-only by default, grounding-required for GENERATION_GROUNDING, and the
    method scope carried as a tuple (ALL -> None)."""
    assert Query(text="x").permitted_filter() == ChunkFilter()  # EXECUTION + ALL -> available-only
    assert Query(text="x", purpose=Purpose.GENERATION_GROUNDING).permitted_filter().require_grounding
    assert Query(text="x").permitted_filter().method_scope is None  # ALL -> unrestricted
    assert Query(text="x", scope=[MethodRef("M1")]).permitted_filter().method_scope == (MethodRef("M1"),)
    assert Query(text="x").permitted_filter().license_classes is None  # the baseline applies no class filter


def test_default_license_policy_maps_purpose_to_classes():
    """The DefaultLicensePolicy maps purpose -> permitted license classes (ModusQ §5.6 default), never
    permitting third_party_copyrighted, and folds in the query's grounding/scope via permitted_filter."""
    pol = DefaultLicensePolicy(DefaultLicensePolicy.Config())
    f = pol.filter_for(Query(text="x"))
    assert "third_party_copyrighted" not in f.license_classes  # the safety net
    assert "public_domain" in f.license_classes and "customer_licensed" in f.license_classes
    assert pol.filter_for(Query(text="x", purpose=Purpose.GENERATION_GROUNDING)).require_grounding


async def test_default_license_policy_excludes_third_party_copyrighted(repo):
    """End-to-end: with the default policy (engine built from Settings), a third_party_copyrighted chunk is
    never returned, while a shippable-class chunk is."""
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))
    _, (a, b) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="open content", chunk_index=0, total_chunks=2,
                  metadata={"license_class": "public_domain"}),
            Chunk(parent_doc_id="s1", content="copyrighted content", chunk_index=1, total_chunks=2,
                  metadata={"license_class": "third_party_copyrighted"}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=a, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=b, vector=[0.9, 0.1, 0.0]),
    ])
    settings = Settings(_env_file=None)  # default components -> the default_license policy applies
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)), settings=settings)
    assert [r.chunk_id for r in await engine.search(Query(text="content", top_k=5))] == [a]


async def _index_tree(repo):
    """Index a 1-level auto-merging tree: a section parent (ordinal 0, NOT embedded) + two leaf children
    (ordinals 1/2, ``parent_ordinal=0``, embedded). Returns ``(parent_id, leaf1_id, leaf2_id)``."""
    await repo.write_index_meta(IndexMeta.build(_FakeEmbedder()))
    _, (parent, leaf1, leaf2) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="Safety: wear PPE and inspect the tank.", chunk_index=0,
                  provenance=ChunkProvenance(content_hash="p", header_path=["Safety"], level=1)),
            Chunk(parent_doc_id="s1", content="wear PPE", chunk_index=1, metadata={"parent_ordinal": 0}),
            Chunk(parent_doc_id="s1", content="inspect the tank", chunk_index=2,
                  metadata={"parent_ordinal": 0}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=leaf1, vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=leaf2, vector=[0.0, 1.0, 0.0]),
    ])
    return parent, leaf1, leaf2


def _merge_pipeline(**merger):
    return RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}], fuser={"class_name": "identity"},
            merger={"class_name": "auto_merge", **merger},
        )
    )


async def test_auto_merge_collapses_siblings_into_parent(repo):
    parent, _leaf1, _leaf2 = await _index_tree(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 1.0, 0.0)))
    results = await _merge_pipeline().search(Query(text="ppe tank", top_k=5), ctx)
    # Both leaves retrieved and share a parent -> collapsed into the one hydrated section parent.
    assert [r.chunk_id for r in results] == [parent]
    assert results[0].text == "Safety: wear PPE and inspect the tank."
    assert results[0].provenance.header_path == ["Safety"]  # the parent's provenance, not a leaf's
    assert "dense" in results[0].component_scores  # merged from the children


async def test_auto_merge_threshold_not_met_keeps_leaves(repo):
    parent, leaf1, leaf2 = await _index_tree(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 1.0, 0.0)))
    # min_siblings=3 but only two leaves are retrieved -> no merge.
    results = await _merge_pipeline(min_siblings=3).search(Query(text="ppe tank", top_k=5), ctx)
    assert {r.chunk_id for r in results} == {leaf1, leaf2}
    assert parent not in {r.chunk_id for r in results}


async def test_auto_merger_keeps_leaves_when_parent_unavailable():
    """Unit: a parent that is itself unavailable is skipped — its leaves stay (filter already ran)."""
    def _leaf(cid, score):
        return RetrievalResult(
            chunk_id=cid, text=cid, score=score, component_scores={"dense": score},
            document_id="s1", source_kind="document", standard_id=None, locator=None,
            license_class="public_domain",
            provenance=ChunkProvenance(content_hash=cid, parent_chunk_id="p"),
        )

    class _FakeStore:
        async def hydrate(self, ids):
            return [ChunkRecord(
                chunk_id="p", text="parent", document_id="s1", source_kind="document",
                standard_id=None, locator=None, license_class="public_domain", available=False,
            )]

    out = await AutoMerger(AutoMerger.Config()).merge(
        [_leaf("l1", -1.0), _leaf("l2", -2.0)], RetrievalContext(store=_FakeStore(), embedder=None)
    )
    assert [r.chunk_id for r in out] == ["l1", "l2"]


def _result(cid, score, **scores):
    return RetrievalResult(
        chunk_id=cid, text=cid, score=score, component_scores={"dense": score, **scores},
        document_id="s1", source_kind="document", standard_id=None, locator=None,
        license_class="public_domain",
    )


async def test_cross_encoder_reranker_caps_the_shortlist_and_tails_the_rest():
    """P4: only the top-N incoming candidates are cross-encoder scored (bounds the CPU cost at corpus
    scale); the tail keeps its first-pass order below the reranked shortlist."""

    class _CE:
        def __init__(self):
            self.seen = None

        def score(self, query, texts):
            self.seen = list(texts)
            return [0.1, 0.9][: len(texts)]  # second shortlist entry wins

    ce = _CE()
    ctx = RetrievalContext(store=None, embedder=None, cross_encoder=ce)
    results = [_result("a", -1.0), _result("b", -2.0), _result("c", -3.0)]
    out = await CrossEncoderReranker(CrossEncoderReranker.Config(top_n=2)).rerank(
        Query(text="q"), results, ctx
    )
    assert ce.seen == ["a", "b"]  # only the capped shortlist reaches the model (text == cid here)
    assert [r.chunk_id for r in out] == ["b", "a", "c"]  # reranked shortlist first, tail unchanged
    assert "cross_encoder" not in out[2].component_scores  # the tail is never scored


async def test_cross_encoder_reranker_rescores_and_reorders():
    """Unit: the cross-encoder score becomes the new score (re-ordering); first-pass scores are kept."""
    ce = _FakeCrossEncoder({"a": 0.2, "b": 0.95})
    ctx = RetrievalContext(store=None, embedder=None, cross_encoder=ce)
    out = await CrossEncoderReranker(CrossEncoderReranker.Config()).rerank(
        Query(text="q"), [_result("a", -1.0), _result("b", -2.0)], ctx
    )
    assert [r.chunk_id for r in out] == ["b", "a"]  # reordered by cross-encoder score, not dense rank
    assert out[0].score == 0.95
    assert out[0].component_scores == {"dense": -2.0, "cross_encoder": 0.95}  # original kept + rerank added


async def test_cross_encoder_reranker_requires_model_in_context():
    with pytest.raises(RetrievalError, match="no cross-encoder"):
        await CrossEncoderReranker(CrossEncoderReranker.Config()).rerank(
            Query(text="q"), [_result("a", -1.0)], RetrievalContext(store=None, embedder=None)
        )


def test_rrf_fusion_tie_breaks_by_chunk_id():
    """A fused-score tie must break by ``chunk_id`` ascending (ModusQ §5.5/§9), not by retriever /
    insertion order — the determinism the future C++ parity contract relies on."""
    # 'b' and 'a' earn identical RRF scores (each is rank 1 in one retriever, rank 2 in the other), but
    # 'b' is inserted first (it heads the dense list). The tie-break must still order 'a' before 'b'.
    per_retriever = {
        "dense": [Candidate("b", rank=1, raw_score=0.9), Candidate("a", rank=2, raw_score=0.5)],
        "sparse": [Candidate("a", rank=1, raw_score=8.0), Candidate("b", rank=2, raw_score=7.0)],
    }
    hits = RRFFuser(RRFFuser.Config()).fuse(per_retriever)
    assert hits[0].score == hits[1].score  # genuinely tied -> only the tie-break decides the order
    assert [h.chunk_id for h in hits] == ["a", "b"]


def test_rrf_weights_tilt_the_fusion():
    """S1 (weighted RRF): a per-retriever weight scales that arm's rank contributions — the
    sparse-weighted hybrid promotes what the sparse arm ranked first, without touching the rank core."""
    per_retriever = {
        "dense": [Candidate("a", rank=1, raw_score=0.9), Candidate("b", rank=2, raw_score=0.5)],
        "sparse": [Candidate("b", rank=1, raw_score=8.0), Candidate("a", rank=2, raw_score=7.0)],
    }
    unweighted = RRFFuser(RRFFuser.Config()).fuse(per_retriever)
    assert [h.chunk_id for h in unweighted] == ["a", "b"]  # genuinely tied -> chunk_id asc
    weighted = RRFFuser(RRFFuser.Config(weights={"sparse": 2.0})).fuse(per_retriever)
    assert [h.chunk_id for h in weighted] == ["b", "a"]  # sparse's rank-1 ('b') now wins the tie
    assert weighted[0].score == pytest.approx(2.0 / 61 + 1.0 / 62)  # weighted sparse + unit dense


def test_rrf_unlisted_retrievers_weigh_one():
    """Weights apply only to the named keys; every other retriever contributes at 1.0 — a partial map
    leaves the unnamed arms exactly as unweighted RRF scores them."""
    per_retriever = {"dense": [Candidate("a", rank=1, raw_score=0.9)]}
    hits = RRFFuser(RRFFuser.Config(weights={"sparse": 3.0})).fuse(per_retriever)
    assert hits[0].score == pytest.approx(1.0 / 61)


def test_rrf_rejects_non_positive_weights():
    """A zero/negative weight would silently corrupt the ranking — refuse it at config time."""
    with pytest.raises(ValueError):
        RRFFuser.Config(weights={"sparse": 0.0})


def test_identity_fusion_orders_best_first():
    """The identity passthrough returns best-first by score (= -rank) with the same deterministic order,
    independent of the order candidates arrive in."""
    per_retriever = {
        "dense": [Candidate("z", rank=2, raw_score=0.1), Candidate("a", rank=1, raw_score=0.9)]
    }
    hits = IdentityFuser(IdentityFuser.Config()).fuse(per_retriever)
    assert [h.chunk_id for h in hits] == ["a", "z"]  # rank 1 ('a') before rank 2 ('z'), not input order


async def test_create_rejects_a_single_injected_dependency(repo):
    """``create`` shares a store/embedder when *both* are injected; one alone is a wiring error."""
    settings = Settings(_env_file=None, EMBEDDING_DIMENSION=3)
    with pytest.raises(ValueError, match="both repository and embedder"):
        await RetrievalEngine.create(settings, repository=repo)  # embedder omitted
    with pytest.raises(ValueError, match="both repository and embedder"):
        await RetrievalEngine.create(settings, embedder=_FakeEmbedder())  # repository omitted


async def test_create_without_injection_builds_its_own_store(tmp_path):
    """With neither injected, ``create`` builds the repository + embedder from Settings, then validates via
    ``open`` (here the fresh store has no index yet, so it refuses) — exercising the build branch."""
    settings = Settings(
        _env_file=None, EMBEDDING_DIMENSION=3, embedding={"provider": "hash"},
        database={"document_url": f"sqlite:///{tmp_path}/x.db"},
    )
    with pytest.raises(RetrievalError, match="not been built"):
        await RetrievalEngine.create(settings)


async def test_explain_records_the_pipeline_trace(repo):
    cids = await _index(repo)
    engine = await RetrievalEngine.open(repo, _FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    trace = await engine.explain(Query(text="tank inspection", top_k=2))
    assert [r.chunk_id for r in trace.results] == cids  # trace.results == what search returns
    names = [s.name for s in trace.stages]
    assert names[0] == "fused" and names[-1] == "final"  # the recorded stages
    assert trace.per_retriever  # per-retriever candidates recorded
    trace2 = await engine.explain_text("tank inspection", top_k=2)  # the string convenience
    assert [r.chunk_id for r in trace2.results] == cids


async def test_pipeline_trace_records_merged_and_reranked_stages(repo):
    """A trace through a full pipeline (merger + reranker) records every stage — covering the trace branches
    for fused / merged / reranked / final."""
    await _index_tree(repo)
    pipe = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}], fuser={"class_name": "identity"},
            merger={"class_name": "auto_merge"}, reranker={"class_name": "cross_encoder"},
        )
    )
    ce = _FakeCrossEncoder({"Safety: wear PPE and inspect the tank.": 0.9})  # the merged parent's text
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 1.0, 0.0)), cross_encoder=ce)
    trace = SearchTrace(query=Query(text="ppe tank", top_k=5))
    await pipe.search(Query(text="ppe tank", top_k=5), ctx, trace)
    names = [s.name for s in trace.stages]
    assert names == ["fused", "merged", "reranked", "final"]  # every stage snapshot recorded


async def test_pipeline_reranks_with_cross_encoder(repo):
    cids = await _index(repo)  # cids[0] = "storage tank inspection", cids[1] = "quokka marsupial"
    pipe = RetrievalPipeline(
        RetrievalPipeline.Config(
            retrievers=[{"class_name": "dense"}], fuser={"class_name": "identity"},
            reranker={"class_name": "cross_encoder"},
        )
    )
    ce = _FakeCrossEncoder({"storage tank inspection": 0.1, "quokka marsupial": 0.9})
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)), cross_encoder=ce)
    results = await pipe.search(Query(text="q", top_k=2), ctx)
    # Dense ranks the tank chunk first; the cross-encoder prefers the quokka chunk -> the order flips.
    assert [r.chunk_id for r in results] == [cids[1], cids[0]]
    assert results[0].score == 0.9
    assert set(results[0].component_scores) == {"dense", "cross_encoder"}  # first-pass score kept
