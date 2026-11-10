"""The eval harness: sweep retrieval-pipeline specs over a labeled query set against one index."""

import pytest
from bausatz import ComponentFactory

from tarnrag.contracts import Chunk, Document, Embedding
from tarnrag.eval import (
    EvalQuery,
    EvalSet,
    by_query_type,
    evaluate_pipeline,
    format_reports,
    format_segmented,
    sweep,
)
from tarnrag.retrieval import RetrievalContext, RetrievalPipeline


class _FakeEmbedder:
    """Embeds any query to a fixed 3-d vector (the retrieval tests' fake)."""

    def __init__(self, query_vec=(1.0, 0.0, 0.0)):
        self._vec = list(query_vec)

    def embed_query(self, text):
        return self._vec


async def _index(repo):
    """Ingest two chunks (+ embeddings): cids[0] 'storage tank inspection', cids[1] 'quokka marsupial'."""
    _, cids = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="storage tank inspection", chunk_index=0, total_chunks=2),
            Chunk(parent_doc_id="s1", content="quokka marsupial", chunk_index=1, total_chunks=2),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=cids[0], vector=[1.0, 0.0, 0.0]),
        Embedding(chunk_id=cids[1], vector=[0.0, 1.0, 0.0]),
    ])
    return cids


_DENSE = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]}
_SPARSE = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "sparse"}],
    "fuser": {"class_name": "identity"},
}


async def test_sweep_scores_each_pipeline_spec(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    evalset = EvalSet([EvalQuery(text="tank inspection", relevant=["tank"])])

    reports = await sweep({"dense": _DENSE, "sparse": _SPARSE}, ctx, evalset, k=5)

    assert set(reports) == {"dense", "sparse"}
    for report in reports.values():
        assert report.n == 1 and report.k == 5
        assert report.hit_at_k == 1.0  # both find the tank chunk for a "tank" query
        assert report.mrr == 1.0  # and rank it first
    table = format_reports(reports)
    assert "dense" in table and "sparse" in table and "hit@k" in table


async def test_evaluate_pipeline_scores_a_miss(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    # Gold phrase appears in no chunk -> nothing relevant -> all metrics zero.
    evalset = EvalSet([EvalQuery(text="tank", relevant=["helicopter"])])
    pipeline = ComponentFactory.get().create_as(_DENSE, RetrievalPipeline)
    report = await evaluate_pipeline(pipeline, ctx, evalset, k=5)
    assert report.hit_at_k == 0.0 and report.mrr == 0.0 and report.ndcg_at_k == 0.0
    assert report.per_query[0].relevances and not any(report.per_query[0].relevances)


def test_evalset_loads_query_type():
    es = EvalSet.from_records([
        {"text": "q1", "relevant": ["a"], "query_type": "semantic"},
        {"text": "q2", "relevant": ["b"]},  # unlabeled -> ""
    ])
    assert es.queries[0].query_type == "semantic"
    assert es.queries[1].query_type == ""


async def test_segments_metrics_by_query_type(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder())
    evalset = EvalSet([
        EvalQuery(text="tank inspection", relevant=["tank"], query_type="hits"),     # sparse finds the tank chunk
        EvalQuery(text="quokka marsupial", relevant=["tank"], query_type="misses"),  # sparse finds quokka -> no 'tank'
    ])
    report = await evaluate_pipeline(
        ComponentFactory.get().create_as(_SPARSE, RetrievalPipeline), ctx, evalset, k=5
    )
    seg = by_query_type(report)
    assert set(seg) == {"hits", "misses"}
    assert seg["hits"].hit_at_k == 1.0 and seg["misses"].hit_at_k == 0.0  # the split the segmentation reveals
    assert seg["hits"].n == 1 and seg["misses"].n == 1
    assert report.hit_at_k == 0.5  # the overall report is the mean over both types


_ROUTED = {
    "class_name": "routing_retrieval_pipeline",
    # The eval supplies query_type, so the router dispatches on those labels and skips its classifier.
    "routes": {"lexical": _SPARSE, "semantic": _DENSE},
    "default": _DENSE,
}


async def test_router_dispatches_per_type_and_beats_either_single_pipeline(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder(query_vec=(1.0, 0.0, 0.0)))
    evalset = EvalSet([
        # sparse finds the quokka chunk; dense's top-1 is the (fixed-vector) tank chunk -> dense misses.
        EvalQuery(text="quokka", relevant=["quokka"], query_type="lexical"),
        # dense's top-1 tank chunk contains "inspection"; sparse has no lexical overlap -> sparse misses.
        EvalQuery(text="checking equipment condition", relevant=["inspection"], query_type="semantic"),
    ])
    reports = await sweep({"dense": _DENSE, "sparse": _SPARSE, "routed": _ROUTED}, ctx, evalset, k=1)
    routed, dense, sparse = (by_query_type(reports[n]) for n in ("routed", "dense", "sparse"))

    # Each labeled type is dispatched to its configured method, so per type the router == that method.
    assert routed["lexical"].hit_at_k == sparse["lexical"].hit_at_k == 1.0
    assert routed["semantic"].hit_at_k == dense["semantic"].hit_at_k == 1.0
    # Picking the right method per type beats either single pipeline overall (each wins only one type).
    assert reports["routed"].hit_at_k == 1.0
    assert reports["dense"].hit_at_k == 0.5 and reports["sparse"].hit_at_k == 0.5


async def test_format_segmented_table(repo):
    await _index(repo)
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder())
    evalset = EvalSet([
        EvalQuery(text="tank inspection", relevant=["tank"], query_type="lexical"),
        EvalQuery(text="quokka marsupial", relevant=["quokka"], query_type="lexical"),
    ])
    reports = await sweep({"dense": _DENSE, "sparse": _SPARSE}, ctx, evalset, k=5)
    table = format_segmented(reports, metric="hit_at_k")
    assert "lexical" in table and "all" in table and "dense" in table and "sparse" in table
    with pytest.raises(ValueError, match="unknown metric"):
        format_segmented(reports, metric="bogus")
