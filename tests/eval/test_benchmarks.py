"""The benchmark harness: the loaders (pure) and an end-to-end run over the real ingest → retrieve →
generate → score loop, driven by a deterministic ``StaticLanguageModel`` + the model-free ``hash`` embedder
(so the whole loop runs offline, no API key and no ONNX download)."""

from pathlib import Path

from tarnrag.core.engine.config import Settings
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.eval.benchmark_runner import (
    MOTHRAG_PUBLISHED,
    build_corpus_index,
    format_comparison,
    format_sweep,
    run_benchmark,
    run_over_corpus,
    sweep_benchmark,
    sweep_over_corpus,
)
from tarnrag.eval.benchmarks import (
    corpus_from_items,
    load_2wiki_hf,
    load_hotpotqa,
    load_hotpotqa_hf,
    load_musique,
    load_musique_hf,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hotpot_sample.json"


def _offline_settings() -> Settings:
    """Settings whose embedder is the model-free ``hash`` backend — the eval loop runs with no ONNX model."""
    return Settings(_env_file=None, embedding={"provider": "hash"})


def test_load_hotpotqa_parses_passages_answer_and_supporting():
    items = load_hotpotqa(FIXTURE)
    assert len(items) == 3
    q0 = items[0]
    assert q0.query.answer == "Eiffel Tower"
    assert q0.query.answer_contains == ["Eiffel Tower"]  # not yes/no -> drives content-hit
    assert [t for t, _ in q0.passages] == ["Eiffel Tower", "Louvre", "Big Ben"]
    assert q0.passages[0][1].startswith("The Eiffel Tower is a wrought-iron")  # paragraph = joined sentences
    assert q0.query.supporting == ["It is the tallest structure in Paris."]  # gold sentence, resolved by idx
    # a yes/no answer carries no content-hit phrases (token-F1/EM still apply)
    assert items[2].query.answer == "yes" and items[2].query.answer_contains == []
    assert len(load_hotpotqa(FIXTURE, limit=1)) == 1  # limit


def test_mothrag_published_numbers_present():
    assert set(MOTHRAG_PUBLISHED) == {"hotpotqa", "2wiki", "musique"}
    assert MOTHRAG_PUBLISHED["hotpotqa"]["f1"] == 0.781


def test_hf_record_normalizes_columns_to_a_bench_item():
    """A HuggingFace hotpot_qa row (column-oriented context/supporting_facts) normalizes to the same
    record the JSON loader uses, then to a BenchItem (no network/`datasets` needed for the conversion)."""
    from tarnrag.eval.benchmarks import _hf_hotpot_record, _hotpot_item

    hf_row = {
        "question": "Are A and B the same?",
        "answer": "yes",
        "context": {"title": ["A", "B"], "sentences": [["a0.", "a1."], ["b0."]]},
        "supporting_facts": {"title": ["A"], "sent_id": [1]},
    }
    record = _hf_hotpot_record(hf_row)
    assert record["context"] == [("A", ["a0.", "a1."]), ("B", ["b0."])]
    assert record["supporting_facts"] == [("A", 1)]

    item = _hotpot_item(record)
    assert item.query.answer == "yes" and item.query.answer_contains == []  # yes/no -> no content-hit
    assert item.query.supporting == ["a1."]  # gold sentence resolved by (title, sent_id)
    assert [t for t, _ in item.passages] == ["A", "B"]


def test_2wiki_hf_record_cleans_messy_encoding():
    """voidful/2WikiMultihopQA wraps titles in literal quotes, JSON-encodes the sentence list as a string,
    and stores sent_id as a string — the cleaner normalizes all three to the _hotpot_item shape."""
    from tarnrag.eval.benchmarks import _2wiki_hf_record, _hotpot_item

    row = {
        "question": "Who directed it?",
        "answer": "Ada",
        "context": [['"Film X"', '["Film X was directed by Ada.", "It came out in 1990."]'], ['"Other"', '["noise."]']],
        "supporting_facts": [['"Film X"', "0"]],
    }
    rec = _2wiki_hf_record(row)
    assert rec["context"] == [("Film X", ["Film X was directed by Ada.", "It came out in 1990."]), ("Other", ["noise."])]
    assert rec["supporting_facts"] == [("Film X", 0)]
    item = _hotpot_item(rec)
    assert item.query.answer == "Ada"
    assert item.query.supporting == ["Film X was directed by Ada."]  # gold sentence resolved


def test_musique_item_carries_aliases_and_abstention():
    from tarnrag.eval.benchmarks import _musique_item

    row = {
        "question": "Who is the spouse?",
        "answer": "Miquette Giraudy",
        "answer_aliases": ["Giraudy"],
        "answerable": True,
        "paragraphs": [
            {"title": "A", "paragraph_text": "Gong is a band.", "is_supporting": True},
            {"title": "B", "paragraph_text": "noise", "is_supporting": False},
        ],
    }
    item = _musique_item(row)
    assert item.query.answer == "Miquette Giraudy" and item.query.answer_aliases == ["Giraudy"]
    assert item.query.answer_contains == ["Miquette Giraudy", "Giraudy"]  # aliases feed content-hit too
    assert item.query.supporting == ["Gong is a band."] and item.query.should_abstain is False
    # unanswerable -> should_abstain
    assert _musique_item({"question": "q", "answer": "", "answerable": False, "paragraphs": []}).query.should_abstain


def test_alias_scoring_takes_the_max():
    """An answer matching an alias (not the primary gold) still scores a perfect F1/EM."""
    from tarnrag.eval.generation import GenerationResult, GenEvalQuery, _score

    q = GenEvalQuery(text="q", answer="Miquette Giraudy", answer_aliases=["Giraudy"])
    scored = _score(q, GenerationResult(answer="Giraudy", abstained=False, grounded=True, evidence=[], proof=[]))
    assert scored.exact_match is True and scored.token_f1 == 1.0


async def test_run_benchmark_end_to_end_offline():
    """Full distractor loop per question (ingest its passages → retrieve → answer → score), over the
    model-free hash embedder + a canned LLM. The canned answer matches q1's gold (perfect F1/EM); others
    differ. Runs in CI — no ONNX model, no API key."""
    items = load_hotpotqa(FIXTURE)
    canned = '{"answer": "Eiffel Tower", "steps": [{"claim": "The tallest structure in Paris.", "cited": [1]}]}'
    report = await run_benchmark(items, StaticLanguageModel(canned), settings=_offline_settings())

    assert report.n == 3
    assert report.per_query[0].exact_match is True and report.per_query[0].token_f1 == 1.0  # q1 == canned
    assert report.per_query[1].exact_match is False  # q2 gold "Rome" != canned
    table = format_comparison({"hotpotqa": report})
    assert "hotpotqa" in table and "0.781" in table  # rendered against MOTHRAG's published F1


async def test_sweep_benchmark_returns_a_report_per_reasoner():
    """The Phase-0 reasoner sweep: each named reasoner is run over the same per-question passages and gets
    its own aggregate report; the table renders against MOTHRAG's reference for the dataset."""
    items = load_hotpotqa(FIXTURE)
    canned = '{"answer": "Eiffel Tower", "steps": [{"claim": "Tallest in Paris.", "cited": [1]}]}'
    reports = await sweep_benchmark(
        items, StaticLanguageModel(canned), reasoners=["single_hop"], settings=_offline_settings()
    )

    assert set(reports) == {"single_hop"}
    assert reports["single_hop"].n == 3
    assert reports["single_hop"].per_query[0].token_f1 == 1.0  # q1 matches the canned answer
    table = format_sweep("hotpotqa", reports)
    assert "single_hop" in table and "0.781" in table  # MOTHRAG reference line for hotpotqa


def test_hf_loaders_map_injected_rows():
    """The HF loaders' mapping / filtering / limit logic, exercised with injected rows (no network / no
    ``datasets``). The ``load_dataset`` call itself is the only integration-only line."""
    hp = load_hotpotqa_hf(
        rows=[{
            "question": "q", "answer": "a",
            "context": {"title": ["T"], "sentences": [["s0.", "s1."]]},
            "supporting_facts": {"title": ["T"], "sent_id": [1]},
        }]
    )
    assert hp[0].query.answer == "a" and hp[0].query.supporting == ["s1."]

    tw = load_2wiki_hf(rows=[{
        "question": "q", "answer": "a",
        "context": [['"T"', '["s0."]']], "supporting_facts": [['"T"', "0"]],
    }])
    assert tw[0].query.supporting == ["s0."]  # the messy 2Wiki encoding is cleaned

    mq_rows = [
        {"question": "q1", "answer": "a1", "answerable": True, "paragraphs": [{"title": "A", "paragraph_text": "p", "is_supporting": True}]},
        {"question": "q2", "answer": "", "answerable": False, "paragraphs": []},
        {"question": "q3", "answer": "a3", "answerable": True, "paragraphs": []},
    ]
    assert [it.query.text for it in load_musique_hf(rows=mq_rows)] == ["q1", "q3"]  # unanswerable filtered
    assert len(load_musique_hf(rows=mq_rows, limit=1)) == 1  # limit


def test_load_musique_file(tmp_path):
    """The MuSiQue JSON-Lines file loader."""
    import json as _json

    rows = [
        {"question": "q1", "answer": "a1", "answerable": True,
         "paragraphs": [{"title": "A", "paragraph_text": "p", "is_supporting": True}]},
        {"question": "q2", "answer": "a2", "answerable": True, "paragraphs": []},
    ]
    path = tmp_path / "musique.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    items = load_musique(str(path))
    assert [it.query.text for it in items] == ["q1", "q2"]
    assert items[0].query.supporting == ["p"]
    assert len(load_musique(str(path), limit=1)) == 1


def test_corpus_from_items_dedups_passages():
    items = load_hotpotqa(FIXTURE)
    corpus = corpus_from_items(items)
    texts = [t for _, t in corpus]
    assert len(texts) == len(set(texts))  # deduped by passage text
    assert set(texts) == {text for it in items for _, text in it.passages}  # the whole union


async def test_build_corpus_index_and_retrieve_only_run(tmp_path):
    """The fullwiki-style path: ingest one shared corpus, then retrieve-only per question (no per-question
    ingest). Offline via the hash embedder + a canned LLM."""
    settings = _offline_settings()
    corpus = corpus_from_items(load_hotpotqa(FIXTURE))
    repo, embedder = await build_corpus_index(corpus, settings, db_path=str(tmp_path / "corpus.db"))
    try:
        items = load_hotpotqa(FIXTURE)
        canned = '{"answer": "Eiffel Tower", "steps": [{"claim": "x", "cited": [1]}]}'
        report = await run_over_corpus(items, StaticLanguageModel(canned), repo, embedder, settings=settings)
        assert report.n == 3 and report.per_query[0].token_f1 == 1.0  # canned matches q1's gold

        reports = await sweep_over_corpus(
            items, StaticLanguageModel(canned), repo, embedder, reasoners=["single_hop"], settings=settings
        )
        assert set(reports) == {"single_hop"} and reports["single_hop"].n == 3
    finally:
        await repo.disconnect()


async def test_corpus_index_is_cached(tmp_path):
    """A second build over the same corpus reuses the index (no re-ingest) — so baseline-vs-bridge runs
    share one (slow) embedding pass."""
    settings = _offline_settings()
    corpus = [("A", "alpha text one"), ("B", "beta text two")]
    db = str(tmp_path / "c.db")
    repo, _ = await build_corpus_index(corpus, settings, db_path=db)
    n1 = len(await repo.list_documents())
    await repo.disconnect()
    repo2, _ = await build_corpus_index(corpus, settings, db_path=db)  # cached -> reuse
    n2 = len(await repo2.list_documents())
    await repo2.disconnect()
    assert n1 == 2 and n2 == 2  # not re-ingested / duplicated


async def test_run_over_corpus_concurrency_preserves_results(tmp_path):
    """Bounded concurrency overlaps the I/O-bound calls without changing the (order-independent) result."""
    settings = _offline_settings()
    corpus = corpus_from_items(load_hotpotqa(FIXTURE))
    repo, embedder = await build_corpus_index(corpus, settings, db_path=str(tmp_path / "c.db"))
    try:
        items = load_hotpotqa(FIXTURE)
        lm = StaticLanguageModel('{"answer": "Eiffel Tower", "steps": [{"claim": "x", "cited": [1]}]}')
        seq = await run_over_corpus(items, lm, repo, embedder, settings=settings, concurrency=1)
        par = await run_over_corpus(items, lm, repo, embedder, settings=settings, concurrency=4)
        assert seq.n == par.n == 3
        assert [r.token_f1 for r in seq.per_query] == [r.token_f1 for r in par.per_query]  # same per-item
    finally:
        await repo.disconnect()


def test_hybrid_retrieval_spec_builds():
    from bausatz import ComponentFactory

    from tarnrag.eval.benchmark_runner import HYBRID_RETRIEVAL
    from tarnrag.retrieval.pipeline.pipeline import RetrievalPipeline

    pipeline = ComponentFactory.get().create_as(HYBRID_RETRIEVAL, RetrievalPipeline)
    assert pipeline is not None  # dense + sparse retrievers, RRF fuser — a valid retrieval pipeline


def test_reasoner_spec_grounding_override():
    from tarnrag.eval.benchmark_runner import _reasoner_spec

    assert _reasoner_spec("decomposition", None) == {
        "class_name": "generation_pipeline", "reasoner": {"class_name": "decomposition"}
    }
    # grounding overrides only the grounded_retrieval reasoner's checker
    assert _reasoner_spec("grounded_retrieval", "llm_grounding")["reasoner"]["grounding_checker"] == {
        "class_name": "llm_grounding"
    }
    assert "grounding_checker" not in _reasoner_spec("decomposition", "llm_grounding")["reasoner"]
    assert "grounding_checker" not in _reasoner_spec("grounded_retrieval", None)["reasoner"]


async def test_safe_answer_guards_failures():
    """A per-question failure yields an empty result (scored a miss) instead of crashing the sweep; a
    success passes through."""
    from tarnrag.eval.benchmark_runner import _safe_answer
    from tarnrag.generation import GenerationResult

    async def boom() -> GenerationResult:
        raise RuntimeError("llm down")

    async def ok() -> GenerationResult:
        return GenerationResult(answer="hi")

    assert (await _safe_answer(boom())).answer == ""
    assert (await _safe_answer(ok())).answer == "hi"
