"""The TAT-QA layout-aware retrieval eval: the pure loader/render logic, and an end-to-end source-hit run
over the real ingest → retrieve loop driven by the model-free ``hash`` embedder (offline, no network)."""

import json

from tarnrag.core.engine.config import Settings
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.eval.layout import (
    NumericReport,
    build_tatqa_index,
    format_attribution,
    format_numeric,
    format_source_hit,
    load_tatqa,
    load_tatqa_numeric,
    render_table,
    tatqa_attribution,
    tatqa_numeric,
    tatqa_source_hit,
)

ROWS = [
    {
        "table": {"uid": "t1", "table": [["", "2019", "2018"], ["Revenue", "$10", "$8"]]},
        "paragraphs": [
            {"uid": "p1", "order": 1, "text": "Revenue grew strongly in 2019."},
            {"uid": "p2", "order": 2, "text": "Operating costs were flat year over year."},
        ],
        "questions": [
            {"question": "What was revenue in 2019?", "answer": ["$10"], "answer_type": "span",
             "answer_from": "table", "rel_paragraphs": []},
            {"question": "How did revenue change?", "answer": ["grew"], "answer_type": "span",
             "answer_from": "text", "rel_paragraphs": ["1"]},
            {"question": "What is total revenue?", "answer": ["18"], "answer_type": "arithmetic",
             "answer_from": "table", "rel_paragraphs": []},  # dropped: not extractive
        ],
    },
]


def test_render_table_linearizes_cells():
    assert render_table([["a", "b"], ["c", "d"]]) == "a | b\nc | d"


def test_load_tatqa_filters_arithmetic_and_labels_sources():
    import json

    corpus, queries = load_tatqa(ROWS)
    by_uid = {uid: (content, kind) for uid, content, kind in corpus}
    assert set(by_uid) == {"t1", "p1", "p2"}  # table + 2 paragraphs, deduped by uid
    content, kind = by_uid["t1"]
    assert kind == "table"  # the native table_json path — chunks carry a real Table
    assert json.loads(content) == [["", "2019", "2018"], ["Revenue", "$10", "$8"]]  # the JSON grid
    assert by_uid["p1"] == ("Revenue grew strongly in 2019.", "text")
    assert len(queries) == 2  # the arithmetic question is filtered out
    by_seg = {q.answer_from: q for q in queries}
    assert by_seg["table"].gold_source_ids == ["t1"]
    assert by_seg["text"].gold_source_ids == ["p1"]  # rel_paragraph order "1" -> p1


def test_load_tatqa_limit_caps_records():
    corpus, _ = load_tatqa(ROWS + ROWS, limit=1)
    assert {uid for uid, _, _ in corpus} == {"t1", "p1", "p2"}  # only the first record


async def test_source_hit_runs_over_the_index(tmp_path):
    settings = Settings(_env_file=None, embedding={"provider": "hash"})
    corpus, queries = load_tatqa(ROWS)
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=str(tmp_path / "t.db"))
    try:
        # The native path: the table element's chunk carries the persisted Table (cells + headers).
        [chunk] = await repo.get_chunks_by_document("t1")
        assert chunk.provenance is not None and chunk.provenance.table is not None
        assert chunk.provenance.table.cell_at(1, 1).text == "$10"
        assert chunk.content == " | 2019 | 2018\nRevenue | $10 | $8"  # stored/BM25 text: the grid

        report = await tatqa_source_hit(queries, repo, embedder, settings=settings, k=3, concurrency=2)
        assert report.n == 2 and 0.0 <= report.source_hit <= 1.0
        assert set(report.by_segment) == {"table", "text"}  # segmented by answer_from
        assert report.by_segment["table"][0] == 1 and report.by_segment["text"][0] == 1  # one query each
        assert "OVERALL" in format_source_hit(report, tag="[DENSE]")
    finally:
        await repo.disconnect()


async def test_attribution_runs_and_segments(tmp_path):
    settings = Settings(_env_file=None, embedding={"provider": "hash"})
    corpus, queries = load_tatqa(ROWS)
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=str(tmp_path / "t.db"))
    try:
        # the canned answer matches the table question's gold ($10); single_hop reads it, llm_grounding judges
        llm = StaticLanguageModel('{"answer": "$10", "steps": [{"claim": "Revenue was $10", "cited": [1]}]}')
        overall, by_seg = await tatqa_attribution(
            queries, llm, repo, embedder, settings=settings, table_view="structured", concurrency=2
        )
        assert overall.n == 2 and set(by_seg) == {"table", "text"}  # segmented by answer_from
        assert 0.0 <= overall.grounded_rate <= 1.0 and 0.0 <= overall.token_f1 <= 1.0
        assert "OVERALL" in format_attribution(overall, by_seg)
    finally:
        await repo.disconnect()


NUMERIC_ROWS = [
    {
        "table": {"uid": "t1", "table": [["", "2019", "2018"], ["Revenue", "$10", "$8"]]},
        "paragraphs": [{"uid": "p1", "order": 1, "text": "Revenue grew."}],
        "questions": [
            {"question": "What was the change in revenue between 2018 and 2019?",
             "answer": 2, "scale": "million", "answer_type": "arithmetic", "answer_from": "table",
             "rel_paragraphs": []},
            {"question": "What was revenue in 2019?", "answer": ["$10"], "answer_type": "span",
             "answer_from": "table", "rel_paragraphs": []},  # not arithmetic -> not loaded
            {"question": "What is the sum of things?", "answer": "n/a", "answer_type": "arithmetic",
             "answer_from": "table", "rel_paragraphs": []},  # unparseable gold -> skipped
            {"question": "What was the change in revenue between 2016 and 2017?",
             "answer": 1, "scale": "million", "answer_type": "arithmetic", "answer_from": "table",
             "rel_paragraphs": []},  # years absent from the table -> table_lookup abstains
        ],
    },
]


def test_load_tatqa_numeric_keeps_parseable_arithmetic_golds():
    queries = load_tatqa_numeric(NUMERIC_ROWS)
    assert len(queries) == 2  # span + unparseable-gold questions dropped
    assert queries[0].gold == 2.0 and queries[0].scale == "million"


def test_numeric_report_rates_guard_zero_division():
    empty = NumericReport(n=0, attempted=0, correct=0)
    assert empty.attempted_rate == 0.0 and empty.em_attempted == 0.0 and empty.em_overall == 0.0
    assert "n=0" in format_numeric(empty)


async def test_tatqa_numeric_runs_llm_free_over_the_index(tmp_path):
    """End to end, no LLM anywhere: native table ingest -> hybrid retrieval (pinned) ->
    table_lookup (pinned, no fallback) -> numeric EM against the gold."""
    from tarnrag.core.engine.config import RETRIEVAL_PIPELINE

    settings = Settings(_env_file=None, embedding={"provider": "hash"})
    settings.components[RETRIEVAL_PIPELINE] = {  # pinned — never inherited
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
        "fuser": {"class_name": "rrf"},
    }
    corpus, _ = load_tatqa(NUMERIC_ROWS)
    queries = load_tatqa_numeric(NUMERIC_ROWS)
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=str(tmp_path / "n.db"))
    try:
        spec = {"class_name": "table_lookup", "fallback": None, "top_k": 3, "row_coverage": 0.5}
        report = await tatqa_numeric(
            queries, repo, embedder, settings=settings, reasoner_spec=spec, k=3, concurrency=2
        )
        # q1: 10 - 8 = 2, exactly; q2: 2016/2017 not in the table -> abstained = unattempted.
        assert report.n == 2 and report.attempted == 1 and report.correct == 1
        assert "EM overall" in format_numeric(report, tag="[HYBRID]")

        # A reader leg whose answer isn't a number counts as attempted-but-wrong, never a crash.
        reply = StaticLanguageModel(lambda _: json.dumps({"answer": "roughly ten", "steps": []}))
        report = await tatqa_numeric(
            queries, repo, embedder, settings=settings,
            reasoner_spec={"class_name": "single_hop", "top_k": 3}, llm=reply, k=3, concurrency=2,
        )
        assert report.n == 2 and report.attempted == 2 and report.correct == 0
    finally:
        await repo.disconnect()
