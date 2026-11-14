"""The GroundingChecker seam: the heuristic (3-band overlap) + LLM checkers, and the cascading composite."""

import json

from bausatz import ComponentFactory

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import (
    GenerationContext,
    GroundingChecker,
    HeuristicGroundingChecker,
    LLMGroundingChecker,
    ReasonedAnswer,
    ReasonedStep,
    Verdict,
)


def _reasoned(steps, evidence):
    return ReasonedAnswer(answer="ans", steps=steps, evidence=evidence)


def _heuristic(**cfg):
    return HeuristicGroundingChecker(HeuristicGroundingChecker.Config(**cfg))


def _llm(reply):
    return GenerationContext(None, StaticLanguageModel(reply))


def _counting_llm(reply):
    """A context whose LM records each call — to assert the cascade skips it when nothing is uncertain."""
    calls: list[int] = []

    def _reply(_prompt):
        calls.append(1)
        return reply

    return GenerationContext(None, StaticLanguageModel(_reply)), calls


def _cascade(*child_specs):
    spec = {"class_name": "cascading_grounding", "checkers": list(child_specs)}
    return ComponentFactory.get().create_as(spec, GroundingChecker)


# ---------------- heuristic (three bands) ----------------


async def test_heuristic_grounds_a_fully_supported_claim(make_result):
    ev = [make_result("a", "Coatings resist corrosion on steel tanks.")]
    reasoned = _reasoned([ReasonedStep(claim="coatings resist corrosion", cited=[0])], ev)
    assert await _heuristic().check(reasoned, None) == [Verdict.GROUNDED]


async def test_heuristic_flags_a_disjoint_claim_ungrounded(make_result):
    ev = [make_result("a", "Coatings resist corrosion."), make_result("b", "Quokkas are marsupials.")]
    reasoned = _reasoned([ReasonedStep(claim="quokkas are marsupials", cited=[0])], ev)  # cites coatings
    assert await _heuristic().check(reasoned, None) == [Verdict.UNGROUNDED]


async def test_heuristic_is_uncertain_in_the_middle_band(make_result):
    ev = [make_result("a", "Coatings resist corrosion.")]
    # "coatings prevent corrosion" shares 2 of its 3 content words with the passage -> overlap 0.667
    reasoned = _reasoned([ReasonedStep(claim="coatings prevent corrosion", cited=[0])], ev)
    assert await _heuristic().check(reasoned, None) == [Verdict.UNCERTAIN]  # default band (0.1, 0.7)
    assert await _heuristic(high_overlap=0.6).check(reasoned, None) == [Verdict.GROUNDED]  # 0.667 >= 0.6
    assert await _heuristic(low_overlap=0.7).check(reasoned, None) == [Verdict.UNGROUNDED]  # 0.667 <= 0.7


async def test_heuristic_claim_with_no_content_words_is_ungrounded(make_result):
    ev = [make_result("a", "anything at all")]
    reasoned = _reasoned([ReasonedStep(claim="the a of", cited=[0])], ev)  # all stopwords / single chars
    assert await _heuristic().check(reasoned, None) == [Verdict.UNGROUNDED]


# ---------------- LLM (decisive) ----------------


async def test_llm_parses_decisive_verdicts(make_result):
    ev = [make_result("a", "x"), make_result("b", "y")]
    reasoned = _reasoned([ReasonedStep(claim="c1", cited=[0]), ReasonedStep(claim="c2", cited=[1])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(
        reasoned, _llm(json.dumps({"verdicts": [True, False]}))
    )
    assert out == [Verdict.GROUNDED, Verdict.UNGROUNDED]


async def test_llm_is_uncertain_on_bad_json(make_result):
    ev = [make_result("a", "x")]
    reasoned = _reasoned([ReasonedStep(claim="c", cited=[0])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(reasoned, _llm("not json at all"))
    assert out == [Verdict.UNCERTAIN]  # undecided, not a pass — the pipeline policy collapses it


async def test_llm_pads_missing_verdicts_as_uncertain(make_result):
    ev = [make_result("a", "x"), make_result("b", "y")]
    reasoned = _reasoned([ReasonedStep(claim="c1", cited=[0]), ReasonedStep(claim="c2", cited=[1])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(
        reasoned, _llm(json.dumps({"verdicts": [False]}))  # only one verdict for two steps
    )
    assert out == [Verdict.UNGROUNDED, Verdict.UNCERTAIN]


def test_llm_reads_quoted_and_wrapped_booleans_strictly():
    """MED-06: ``bool("false")`` is True — a quoted ``"false"`` must not grade a rejected claim GROUNDED."""
    reply = json.dumps({"verdicts": ["false", "True", {"step": 0, "grounded": "false"}, 1, "maybe", None]})
    assert LLMGroundingChecker._parse(reply, 6) == [
        Verdict.UNGROUNDED, Verdict.GROUNDED, Verdict.UNGROUNDED, Verdict.GROUNDED,
        Verdict.UNCERTAIN, Verdict.UNCERTAIN,
    ]


async def test_llm_grades_an_uncited_step_ungrounded_without_trusting_the_reply(make_result):
    ev = [make_result("a", "x")]
    reasoned = _reasoned([ReasonedStep(claim="c1", cited=[0]), ReasonedStep(claim="c2", cited=[])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(
        reasoned, _llm(json.dumps({"verdicts": [True, True]}))
    )
    assert out == [Verdict.GROUNDED, Verdict.UNGROUNDED]


# ---------------- cascade ----------------


async def test_cascade_escalates_only_the_uncertain_steps(make_result):
    ev = [make_result("a", "Coatings resist corrosion.")]
    reasoned = _reasoned(
        [
            ReasonedStep(claim="coatings resist corrosion", cited=[0]),  # overlap 1.0 -> GROUNDED (heuristic)
            ReasonedStep(claim="coatings rust", cited=[0]),  # overlap 0.5 -> UNCERTAIN -> escalates
        ],
        ev,
    )
    ctx, calls = _counting_llm(json.dumps({"verdicts": [False]}))  # resolves the one escalated step
    cascade = _cascade({"class_name": "heuristic_grounding"}, {"class_name": "llm_grounding"})
    out = await cascade.check(reasoned, ctx)
    assert out == [Verdict.GROUNDED, Verdict.UNGROUNDED]
    assert len(calls) == 1  # the LLM ran once, only for the uncertain step


async def test_cascade_skips_the_llm_when_the_heuristic_resolves_everything(make_result):
    ev = [make_result("a", "Coatings resist corrosion.")]
    reasoned = _reasoned([ReasonedStep(claim="coatings resist corrosion", cited=[0])], ev)  # -> GROUNDED
    ctx, calls = _counting_llm(json.dumps({"verdicts": [False]}))
    cascade = _cascade({"class_name": "heuristic_grounding"}, {"class_name": "llm_grounding"})
    assert await cascade.check(reasoned, ctx) == [Verdict.GROUNDED]
    assert calls == []  # the expensive checker was never called


async def test_llm_judge_sees_the_configured_table_view():
    """The fact-checker renders cited TABLE chunks per its table_view — structured by default, the raw
    grid when pinned to 'text' — so it judges the same representation the reader read."""
    import json as _json

    from tarnrag.contracts import ChunkProvenance, RetrievalResult, Table, TableCell
    from tarnrag.core.resources.llm import StaticLanguageModel
    from tarnrag.generation.components.grounding import LLMGroundingChecker
    from tarnrag.generation.components.reasoner import ReasonedAnswer, ReasonedStep
    from tarnrag.generation.context import GenerationContext

    table = Table(n_rows=2, n_cols=2, cells=[
        TableCell(id="h", row=0, col=1, is_column_header=True, text="2019"),
        TableCell(id="r", row=1, col=0, is_row_header=True, text="Goodwill"),
        TableCell(id="v", row=1, col=1, text="1,910"),
    ])
    hit = RetrievalResult(
        chunk_id="t", text=" | 2019\nGoodwill | 1,910", score=1.0, component_scores={},
        document_id="d", source_kind="table", standard_id=None, locator=None, license_class="open",
        provenance=ChunkProvenance(content_hash="h", table=table),
    )
    reasoned = ReasonedAnswer("1,910", [ReasonedStep(claim="Goodwill was 1,910 in 2019", cited=[0])], [hit])
    seen = {}

    def reply(prompt):
        seen["user"] = prompt.user
        return _json.dumps({"verdicts": [True]})

    ctx = GenerationContext(retrieval=None, llm=StaticLanguageModel(reply))
    checker = LLMGroundingChecker(LLMGroundingChecker.Config())
    assert await checker.check(reasoned, ctx)  # runs; and the judge saw the structured rendering
    assert "Goodwill \u2014 2019: 1,910" in seen["user"]

    pinned = LLMGroundingChecker(LLMGroundingChecker.Config(table_view="text"))
    await pinned.check(reasoned, ctx)
    assert " | 2019\nGoodwill | 1,910" in seen["user"]  # the raw grid under the pinned baseline view
