"""The answerability gate (PP-2): refuse before the read when the query's exact-match cues aren't in
the evidence; pass through when they are (or when there's nothing checkable)."""

import json

from bausatz import ComponentFactory

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import (
    AnswerabilityGateReasoner,
    GenerationContext,
    GenerationPipeline,
)
from tarnrag.generation.components.reasoner import Reasoner
from tarnrag.retrieval.types import Query


def _gate(**cfg) -> AnswerabilityGateReasoner:
    return ComponentFactory.get().create_as(
        {"class_name": "answerability", **cfg}, AnswerabilityGateReasoner
    )


class _CountingLLM(StaticLanguageModel):
    """A canned reply that counts calls — proves the read was (not) spent."""

    def __init__(self, reply: str):
        super().__init__(reply)
        self.calls = 0

    async def complete(self, prompt):
        self.calls += 1
        return await super().complete(prompt)


async def test_gate_refuses_when_an_identifier_is_not_in_the_evidence(make_result, fake_retrieval):
    """The corpus never mentions E42 — the gate refuses, names the missing cue, spends NO read."""
    llm = _CountingLLM(json.dumps({"answer": "guess"}))
    ctx = GenerationContext(fake_retrieval([make_result("a", "the pump manual covers seals")]), llm)
    out = await _gate().reason(Query(text="pump fails with error E42 whenever E42 appears"), ctx)
    assert out.abstained is True and out.answer.count("E42") == 1  # cue deduped in the refusal
    assert out.steps == [] and [r.chunk_id for r in out.evidence] == ["a"]  # what WAS found rides along
    assert llm.calls == 0  # the read was never spent


async def test_gate_delegates_when_cues_are_covered(make_result, fake_retrieval):
    llm = _CountingLLM(json.dumps({"answer": "reset the controller", "steps": []}))
    results = [make_result("a", "Error E42 means the controller needs a reset.")]
    out = await _gate().reason(Query(text="pump fails with error E42"), ctx := GenerationContext(fake_retrieval(results), llm))
    assert out.abstained is False and out.answer == "reset the controller"
    assert llm.calls == 1  # the wrapped single_hop read ran
    assert ctx.retrieval.seen is not None  # the child re-retrieved through the same port


async def test_gate_passes_through_with_nothing_checkable(make_result, fake_retrieval):
    """No quoted spans, no identifiers — nothing the gate can check; straight to the child (and the
    gate makes no probe retrieval of its own)."""
    retrieval = fake_retrieval([make_result("a", "coatings resist corrosion")])
    llm = _CountingLLM(json.dumps({"answer": "apply a coating", "steps": []}))
    out = await _gate().reason(Query(text="how do i prevent rust on pipes"), GenerationContext(retrieval, llm))
    assert out.abstained is False and llm.calls == 1


async def test_quoted_span_requires_adjacency(make_result, fake_retrieval):
    """A quoted span is covered only as an adjacent token run — scattered words don't count."""
    scattered = [make_result("a", "goodwill rose while impairment fell in 2019")]
    ctx = GenerationContext(fake_retrieval(scattered), _CountingLLM("{}"))
    out = await _gate().reason(Query(text='find the "goodwill impairment" note'), ctx)
    assert out.abstained is True and "goodwill impairment" in out.answer


async def test_pipeline_surfaces_the_gate_abstention(make_result, fake_retrieval):
    """End to end: the pipeline returns abstained=True / grounded=False with an empty proof, and the
    grounding checker never runs (nothing to verify)."""
    spec = {
        "class_name": "generation_pipeline",
        "reasoner": {"class_name": "answerability"},
        "grounding_checker": {"class_name": "llm_grounding"},
    }
    pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
    llm = _CountingLLM("{}")
    ctx = GenerationContext(fake_retrieval([make_result("a", "unrelated text")]), llm)
    res = await pipeline.answer(Query(text="what does §6.4.2 require?"), ctx)
    assert res.abstained is True and res.grounded is False and res.proof == []
    assert "6.4.2" in res.answer  # the refusal names what's missing
    assert llm.calls == 0  # neither the read nor the verification spent a call


async def test_gate_registers_as_a_reasoner():
    gate = ComponentFactory.get().create_as({"class_name": "answerability"}, Reasoner)
    assert isinstance(gate, AnswerabilityGateReasoner)  # composes anywhere a reasoner spec goes
