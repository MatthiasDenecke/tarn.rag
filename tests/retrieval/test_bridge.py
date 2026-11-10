"""The bridge retrieval components: multi-query expansion (MultiQueryRetriever, Phase 2), hypothetical-
answer retrieval (HydeRetriever, S1) + the LLM relevance judge (LlmJudgeReranker). Driven by a
deterministic ``StaticLanguageModel`` + tiny fakes — no network, no key, no model."""

import json

import pytest

from tarnrag.contracts import Candidate, ChunkProvenance, RetrievalResult
from tarnrag.contracts.structure import Span
from tarnrag.core.exceptions import RetrievalError
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.retrieval.components.reranker import LlmJudgeReranker
from tarnrag.retrieval.components.retriever import (
    HydeRetriever,
    MultiQueryRetriever,
    RetrievalContext,
)
from tarnrag.retrieval.types import Query


class _FakeStore:
    """Records dense_knn calls and returns a canned candidate list each time."""

    def __init__(self, results):
        self._results = results
        self.calls = 0

    async def dense_knn(self, vec, k, chunk_filter):
        self.calls += 1
        return list(self._results)


class _SequenceStore:
    """Returns the next canned candidate list per dense_knn call (repeating the last one)."""

    def __init__(self, lists):
        self._lists = [list(candidates) for candidates in lists]
        self.calls = 0

    async def dense_knn(self, vec, k, chunk_filter):
        out = self._lists[min(self.calls, len(self._lists) - 1)]
        self.calls += 1
        return list(out)


class _FakeEmbedder:
    """Zero-vector embedder recording which space (query vs passage) each text was embedded in."""

    def __init__(self):
        self.query_texts = []
        self.passage_texts = []

    def embed_query(self, text):
        self.query_texts.append(text)
        return [0.0, 0.0, 0.0]

    def embed_passages(self, texts):
        self.passage_texts.append(list(texts))
        return [[0.0, 0.0, 0.0] for _ in texts]


def _ctx(store, llm=None, embedder=None):
    return RetrievalContext(store=store, embedder=embedder or _FakeEmbedder(), llm=llm)


def _result(chunk_id, text):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=0.0, component_scores={}, document_id="d",
        source_kind="text", standard_id=None, locator=None, license_class="open",
        provenance=ChunkProvenance(geometry=[Span(start=0, end=len(text))], header_path=[], content_hash="h"),
    )


# ---------------- MultiQueryRetriever ----------------


async def test_multi_query_expands_and_rrf_fuses():
    store = _FakeStore([Candidate("a", 1, 0.1), Candidate("b", 2, 0.2)])
    llm = StaticLanguageModel(json.dumps({"queries": ["rewrite one", "rewrite two"]}))
    out = await MultiQueryRetriever(MultiQueryRetriever.Config(num_variants=2)).retrieve(
        Query(text="q"), _ctx(store, llm)
    )
    assert store.calls == 3  # original + 2 rewrites, each dense-retrieved
    assert [c.chunk_id for c in out] == ["a", "b"]  # deduped + RRF-fused (a out-ranks b)
    assert [c.rank for c in out] == [1, 2]  # re-ranked, 1-based


async def test_multi_query_caps_at_num_variants():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    llm = StaticLanguageModel(json.dumps({"queries": ["r1", "r2", "r3", "r4", "r5"]}))
    await MultiQueryRetriever(MultiQueryRetriever.Config(num_variants=2)).retrieve(Query(text="q"), _ctx(store, llm))
    assert store.calls == 3  # original + only num_variants (2) rewrites


async def test_multi_query_without_llm_is_single_dense():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    out = await MultiQueryRetriever(MultiQueryRetriever.Config()).retrieve(Query(text="q"), _ctx(store, llm=None))
    assert store.calls == 1 and [c.chunk_id for c in out] == ["a"]


async def test_multi_query_unparseable_expansion_uses_original_only():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    await MultiQueryRetriever(MultiQueryRetriever.Config()).retrieve(
        Query(text="q"), _ctx(store, StaticLanguageModel("not json at all"))
    )
    assert store.calls == 1  # no parseable rewrites -> just the original query


# ---------------- HydeRetriever ----------------


async def test_hyde_embeds_hypothesis_as_passage_and_fuses_with_query():
    store = _SequenceStore(
        [
            [Candidate("a", 1, 0.1), Candidate("b", 2, 0.2)],  # the original query's list
            [Candidate("b", 1, 0.3), Candidate("c", 2, 0.4)],  # the hypothesis's list
        ]
    )
    embedder = _FakeEmbedder()
    llm = StaticLanguageModel("Paris is the capital of France.")
    out = await HydeRetriever(HydeRetriever.Config()).retrieve(
        Query(text="q"), _ctx(store, llm, embedder)
    )
    assert embedder.query_texts == ["q"]  # the literal query stays in the query space
    assert embedder.passage_texts == [["Paris is the capital of France."]]  # the fake doc: passage space
    assert store.calls == 2  # one KNN per vector: original + hypothesis
    assert [c.chunk_id for c in out] == ["b", "a", "c"]  # b appears in both lists -> RRF-fused to the top
    assert [c.rank for c in out] == [1, 2, 3]  # re-ranked, 1-based


async def test_hyde_without_llm_is_single_dense():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    embedder = _FakeEmbedder()
    out = await HydeRetriever(HydeRetriever.Config()).retrieve(
        Query(text="q"), _ctx(store, llm=None, embedder=embedder)
    )
    assert store.calls == 1 and [c.chunk_id for c in out] == ["a"]
    assert embedder.passage_texts == []  # no hypothesis -> nothing embedded as a passage


async def test_hyde_blank_completion_degrades_to_dense():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    embedder = _FakeEmbedder()
    await HydeRetriever(HydeRetriever.Config()).retrieve(
        Query(text="q"), _ctx(store, StaticLanguageModel("   \n"), embedder)
    )
    assert store.calls == 1 and embedder.passage_texts == []  # nothing usable -> original query only


async def test_hyde_num_hypotheses_samples_and_fuses_all():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    embedder = _FakeEmbedder()
    llm_calls = []

    def reply(prompt):
        llm_calls.append(prompt.user)
        return f"hypothesis {len(llm_calls)}"

    await HydeRetriever(HydeRetriever.Config(num_hypotheses=3)).retrieve(
        Query(text="q"), _ctx(store, StaticLanguageModel(reply), embedder)
    )
    assert len(llm_calls) == 3  # one sample per hypothesis
    assert embedder.passage_texts == [["hypothesis 1", "hypothesis 2", "hypothesis 3"]]  # one batch
    assert store.calls == 4  # the original query + 3 hypotheses


async def test_hyde_prompt_carries_the_word_cap():
    captured = {}

    def reply(prompt):
        captured["system"], captured["user"] = prompt.system, prompt.user
        return "h"

    await HydeRetriever(HydeRetriever.Config(max_words=25)).retrieve(
        Query(text="what is a tank?"), _ctx(_FakeStore([]), StaticLanguageModel(reply))
    )
    assert "25 words" in captured["system"] and "what is a tank?" in captured["user"]


# ---------------- LlmJudgeReranker ----------------


async def test_llm_judge_rescores_and_reorders():
    results = [_result("a", "alpha"), _result("b", "beta"), _result("c", "gamma")]
    reply = json.dumps(
        {"scores": [{"passage": 1, "score": 1}, {"passage": 2, "score": 9}, {"passage": 3, "score": 5}]}
    )
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel(reply))
    )
    assert [r.chunk_id for r in out] == ["b", "c", "a"]  # by judged score, desc
    assert out[0].score == 9.0 and out[0].component_scores["llm_judge"] == 9.0


async def test_llm_judge_unscored_get_zero_with_deterministic_tiebreak():
    results = [_result("b", "x"), _result("a", "y")]  # neither scored -> 0/0, tie broken by chunk_id asc
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel("{}"))
    )
    assert [r.chunk_id for r in out] == ["a", "b"]


async def test_llm_judge_ignores_out_of_range_and_non_numeric_scores():
    results = [_result("a", "x"), _result("b", "y")]
    reply = json.dumps({"scores": [{"passage": 5, "score": 9}, {"passage": 1, "score": "high"}, {"passage": 2, "score": 3}]})
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel(reply))
    )
    assert [r.chunk_id for r in out] == ["b", "a"]  # b=3, a stayed 0 (non-numeric ignored; 5 out of range)


async def test_llm_judge_caps_at_top_n():
    results = [_result(f"c{i}", f"passage text {i}") for i in range(5)]
    captured = {}

    def judge(prompt):
        captured["user"] = prompt.user
        return json.dumps({"scores": [{"passage": 1, "score": 3}, {"passage": 2, "score": 9}]})

    out = await LlmJudgeReranker(LlmJudgeReranker.Config(top_n=2)).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel(judge))
    )
    assert captured["user"].count("] passage text") == 2  # only the top-2 were sent to the judge
    assert [r.chunk_id for r in out[:2]] == ["c1", "c0"]  # judged: c1=9, then c0=3
    assert [r.chunk_id for r in out[2:]] == ["c2", "c3", "c4"]  # unjudged tail, first-pass order


async def test_llm_judge_requires_an_llm():
    with pytest.raises(RetrievalError, match="no LLM"):
        await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
            Query(text="q"), [_result("a", "x")], _ctx(store=None, llm=None)
        )


async def test_llm_judge_empty_results_is_noop():
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(Query(text="q"), [], _ctx(store=None))
    assert out == []
