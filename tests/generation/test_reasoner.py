"""The SingleHopReasoner: retrieve -> numbered passages -> strict-JSON answer with per-claim citations."""

import json

import pytest

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import SingleHopReasoner
from tarnrag.generation.components.reasoner import Reasoner
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


def _reasoner(**cfg) -> SingleHopReasoner:
    return SingleHopReasoner(SingleHopReasoner.Config(**cfg))


def _ctx(retrieval, reply: str) -> GenerationContext:
    return GenerationContext(retrieval, StaticLanguageModel(reply))


async def test_parses_answer_and_per_claim_citations(make_result, fake_retrieval):
    results = [
        make_result("a", "Coatings resist corrosion.", header_path=["Maint"]),
        make_result("b", "Quokkas are marsupials."),
    ]
    reply = json.dumps(
        {"answer": "Apply a coating.", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    out = await _reasoner().reason(Query(text="prevent rust?"), _ctx(fake_retrieval(results), reply))

    assert out.answer == "Apply a coating."
    assert len(out.steps) == 1
    assert out.steps[0].claim == "coatings resist corrosion"
    assert out.steps[0].cited == [0]  # 1-based passage 1 -> evidence index 0
    assert [r.chunk_id for r in out.evidence] == ["a", "b"]


async def test_falls_back_to_cite_all_on_non_json(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), "Just a plain sentence."))
    assert out.answer == "Just a plain sentence."
    assert len(out.steps) == 1 and out.steps[0].cited == [0, 1]  # the cite-all fallback


async def test_an_explicit_empty_steps_list_cites_nothing(make_result, fake_retrieval):
    """MED-06: ``"steps": []`` is the model citing nothing (e.g. "the passages don't say") — the answer
    must not be dressed up as a step citing every passage."""
    results = [make_result("a", "x"), make_result("b", "y")]
    reply = json.dumps({"answer": "The passages do not say.", "steps": []})
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    assert [(s.claim, s.cited) for s in out.steps] == [("The passages do not say.", [])]


async def test_a_missing_steps_key_still_falls_back_to_cite_all(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), json.dumps({"answer": "ok"})))
    assert out.steps[0].cited == [0, 1]  # a format slip, not a statement — keep the evidence attached


async def test_extracts_json_embedded_in_prose(make_result, fake_retrieval):
    results = [make_result("a", "x")]
    reply = 'Sure!\n{"answer": "ok", "steps": [{"claim": "c", "cited": [1]}]}\nThanks'
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    assert out.answer == "ok" and out.steps[0].cited == [0]


async def test_drops_out_of_range_and_non_int_citations(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    reply = json.dumps({"answer": "ok", "steps": [{"claim": "c", "cited": [2, 5, 0, "x", True, 2]}]})
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    # passage 2 -> idx 1 (kept, deduped); 5 -> out of range; 0 -> idx -1; "x"/True -> not a valid int.
    assert out.steps[0].cited == [1]


async def test_well_formed_answer_without_steps_gets_one_cite_all_step(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), json.dumps({"answer": "ok"})))
    assert len(out.steps) == 1 and out.steps[0].claim == "ok" and out.steps[0].cited == [0, 1]


async def test_read_width_comes_from_config_top_k(make_result, fake_retrieval):
    fr = fake_retrieval([make_result("a", "x")])
    ctx = GenerationContext(fr, StaticLanguageModel(json.dumps({"answer": "ok"})))
    await _reasoner(top_k=3).reason(Query(text="q", top_k=99), ctx)
    assert fr.seen.top_k == 3  # the reasoner's config drives the read, not the caller's query.top_k


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("The answer is Paris", "Paris"),
        ("Answer: 1889", "1889"),
        ('"Eiffel Tower"', "Eiffel Tower"),
        ("Final answer: yes", "yes"),
        ("Eiffel Tower", "Eiffel Tower"),  # already minimal — untouched
        ("Apply a coating.", "Apply a coating."),  # trailing punctuation left to the scorer
        ("answer", "answer"),  # bare word (no 'is'/':') is not stripped
    ],
)
def test_clean_answer_strips_leadins_and_quotes(raw, clean):
    assert Reasoner._clean_answer(raw) == clean


async def test_reason_applies_short_answer_extraction(make_result, fake_retrieval):
    """The reasoner trims the model's lead-in so the answer is the span (what token-F1/EM score)."""
    results = [make_result("a", "x")]
    reply = json.dumps({"answer": "The answer is Paris", "steps": [{"claim": "c", "cited": [1]}]})
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    assert out.answer == "Paris"


async def test_numbered_passages_and_header_label_in_the_prompt(make_result, fake_retrieval):
    results = [make_result("a", "Coatings help.", header_path=["Maint", "Corrosion"])]
    captured = {}

    def capture(prompt):
        captured["user"] = prompt.user
        return json.dumps({"answer": "ok"})

    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(capture))
    await _reasoner().reason(Query(text="how?"), ctx)
    assert "[1] Maint > Corrosion — Coatings help." in captured["user"]
    assert "Question: how?" in captured["user"]


def _table_result():
    """A retrieved TABLE chunk: raw grid as stored text, a real Table on the provenance."""
    from tarnrag.contracts import ChunkProvenance, RetrievalResult, Table, TableCell

    table = Table(n_rows=2, n_cols=2, cells=[
        TableCell(id="h", row=0, col=1, is_column_header=True, text="2019"),
        TableCell(id="r", row=1, col=0, is_row_header=True, text="Goodwill"),
        TableCell(id="v", row=1, col=1, text="1,910"),
    ])
    return RetrievalResult(
        chunk_id="t", text=" | 2019\nGoodwill | 1,910", score=1.0, component_scores={},
        document_id="d", source_kind="table", standard_id=None, locator=None, license_class="open",
        provenance=ChunkProvenance(content_hash="h", table=table),
    )


async def test_reader_sees_the_structured_table_view(fake_retrieval):
    """P1 PR-2: a table chunk is laid before the model as its header-contextualized rendering (values
    bound to headers), not the raw grid."""
    seen = {}

    def reply(prompt):
        seen["user"] = prompt.user
        return json.dumps({"answer": "1,910", "steps": []})

    ctx = GenerationContext(fake_retrieval([_table_result()]), StaticLanguageModel(reply))
    await _reasoner().reason(Query(text="goodwill in 2019?"), ctx)
    assert "Goodwill \u2014 2019: 1,910" in seen["user"]  # rendered from cells
    assert " | 2019" not in seen["user"]  # the raw grid is not shown


async def test_table_view_text_pins_the_raw_grid(fake_retrieval):
    """table_view='text' (the pre-P1 representation — what evals pin for the baseline leg) lays the
    stored grid text before the model unchanged."""
    seen = {}

    def reply(prompt):
        seen["user"] = prompt.user
        return json.dumps({"answer": "1,910", "steps": []})

    ctx = GenerationContext(fake_retrieval([_table_result()]), StaticLanguageModel(reply))
    await _reasoner(table_view="text").reason(Query(text="goodwill in 2019?"), ctx)
    assert " | 2019\nGoodwill | 1,910" in seen["user"]  # the stored text, verbatim
    assert "\u2014 2019: 1,910" not in seen["user"]
