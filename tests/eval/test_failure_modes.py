"""Failure-mode regression sets (PP-4) — one named, deterministic, offline set per production failure
mode, run on every push. Each set pins its full pipeline spec explicitly (a measurement never inherits
a default). Covered here: permission-bound docs, ambiguous acronyms, should-refuse. Table-heavy docs are
covered by the TAT-QA layout eval (on-demand) + the hybrid drift-guard; stale-docs waits on the
effective-date filter axis (roadmap PP-5).
"""

import json

from bausatz import ComponentFactory

from tarnrag.contracts import Chunk, Document, Embedding
from tarnrag.eval import EvalQuery, EvalSet, evaluate_pipeline
from tarnrag.generation import GenerationContext, GenerationPipeline
from tarnrag.retrieval import RetrievalContext, RetrievalPipeline
from tarnrag.retrieval.components.license_policy import DefaultLicensePolicy
from tarnrag.retrieval.types import Query

# Every spec pinned in full — never inherited.
_SPARSE_ONLY = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "sparse"}],
    "fuser": {"class_name": "identity"},
}
_HYBRID = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
    "fuser": {"class_name": "rrf"},
}


class _FakeEmbedder:
    def __init__(self, query_vec=(1.0, 0.0, 0.0)):
        self._vec = list(query_vec)

    def embed_query(self, text):
        return self._vec


def _pipeline(spec) -> RetrievalPipeline:
    return ComponentFactory.get().create_as(dict(spec), RetrievalPipeline)


# ---------------- failure mode: permission-bound documents ----------------


async def test_permission_bound_docs_never_leak(repo):
    """The gold answer lives in a ``third_party_copyrighted`` chunk that is the BEST match on both
    arms — and must still never surface (the §5.6 pre-filter, not post-truncation); the permitted
    runner-up is returned instead, so the caller gets an answer without the leak."""
    _, (forbidden, permitted) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="the exact tank inspection interval is 18 months",
                  chunk_index=0, total_chunks=2,
                  metadata={"license_class": "third_party_copyrighted"}),
            Chunk(parent_doc_id="s1", content="tank inspection guidance summary",
                  chunk_index=1, total_chunks=2,
                  metadata={"license_class": "public_domain"}),
        ],
    )
    await repo.store_embeddings([
        Embedding(chunk_id=forbidden, vector=[1.0, 0.0, 0.0]),  # dense-best
        Embedding(chunk_id=permitted, vector=[0.9, 0.1, 0.0]),
    ])
    ctx = RetrievalContext(
        store=repo, embedder=_FakeEmbedder(),
        license_policy=DefaultLicensePolicy(DefaultLicensePolicy.Config()),
    )
    evalset = EvalSet([EvalQuery(text="tank inspection interval", relevant=["18 months"])])
    report = await evaluate_pipeline(_pipeline(_HYBRID), ctx, evalset, k=5)
    assert report.hit_at_k == 0.0  # the copyrighted gold NEVER surfaces, even as the best match
    results = await _pipeline(_HYBRID).search(Query(text="tank inspection interval", top_k=5), ctx)
    assert [r.chunk_id for r in results] == [permitted]  # the permitted runner-up backfills


# ---------------- failure mode: ambiguous acronyms ----------------


async def test_ambiguous_acronym_resolves_by_context(repo):
    """'PSA' means two different things in the corpus; the context words in the query must pick the
    right expansion deterministically on the lexical arm (BM25 — pinned sparse-only)."""
    _, (medical, broadcast) = await repo.store_document_with_chunks(
        Document(content="d", metadata={"source_id": "s1"}),
        [
            Chunk(parent_doc_id="s1", content="PSA prostate specific antigen reference levels",
                  chunk_index=0, total_chunks=2),
            Chunk(parent_doc_id="s1", content="PSA public service announcement broadcast schedule",
                  chunk_index=1, total_chunks=2),
        ],
    )
    ctx = RetrievalContext(store=repo, embedder=_FakeEmbedder())
    pipeline = _pipeline(_SPARSE_ONLY)
    hits = await pipeline.search(Query(text="PSA antigen levels", top_k=2), ctx)
    assert hits[0].chunk_id == medical  # context words disambiguate
    hits = await pipeline.search(Query(text="PSA broadcast schedule", top_k=2), ctx)
    assert hits[0].chunk_id == broadcast


# ---------------- failure mode: should-refuse (unsupported claims) ----------------


class _FakeRetrieval:
    """A retrieval-port double (local — the generation conftest fixtures aren't visible here)."""

    def __init__(self, results):
        self._results = list(results)

    async def search(self, query):
        return list(self._results)


def _hit(chunk_id: str, text: str):
    from tarnrag.contracts import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=1.0, component_scores={}, document_id="d",
        source_kind="document", standard_id=None, locator=None, license_class="public_domain",
    )


async def test_unsupported_answer_is_refused_not_guessed():
    """The reader fabricates a claim the evidence doesn't support — the (pinned) heuristic grounding
    checker + abstention policy must refuse rather than pass the guess through; a supported answer
    on the same config is NOT refused (no false refusals)."""
    from tarnrag.core.resources.llm import StaticLanguageModel

    spec = {  # pinned in full
        "class_name": "generation_pipeline",
        "reasoner": {"class_name": "single_hop"},
        "grounding_checker": {"class_name": "heuristic_grounding", "high_overlap": 0.7, "low_overlap": 0.1},
        "min_grounded": 1.0,
        "abstain": True,
    }
    pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
    evidence = [_hit("a", "coatings resist corrosion on steel pipes")]

    fabricated = json.dumps(
        {"answer": "42 volts", "steps": [{"claim": "the maximum voltage rating equals exactly 42 volts", "cited": [1]}]}
    )
    res = await pipeline.answer(
        Query(text="max voltage?"), GenerationContext(_FakeRetrieval(evidence), StaticLanguageModel(fabricated))
    )
    assert res.abstained is True and res.grounded is False  # the guess is refused

    supported = json.dumps(
        {"answer": "coatings", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    res = await pipeline.answer(
        Query(text="prevent rust?"), GenerationContext(_FakeRetrieval(evidence), StaticLanguageModel(supported))
    )
    assert res.abstained is False and res.grounded is True  # a supported answer passes
