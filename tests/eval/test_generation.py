"""The generation eval harness: the metric functions + an end-to-end sweep over a labeled set."""

import json

from bausatz import ComponentFactory

from tarnrag.contracts import RetrievalResult
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.eval import (
    GenEvalQuery,
    GenEvalSet,
    citation_coverage,
    content_hit,
    evaluate_generation,
    exact_match,
    format_generation_reports,
    sweep_generation,
    token_f1,
)
from tarnrag.generation import (
    Citation,
    GenerationContext,
    GenerationPipeline,
    GenerationResult,
    ProofStep,
)


def _result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=1.0, component_scores={}, document_id="d",
        source_kind="text", standard_id=None, locator=None, license_class="open",
    )


class _Retrieval:
    """A retrieval-port double returning a fixed result set."""

    def __init__(self, results):
        self._results = results

    async def search(self, query):
        return list(self._results)


def _routed_lm(by_question):
    """An LM that returns a different reply per question (the question is in the reasoner's user prompt)."""

    def _reply(prompt):
        for key, reply in by_question.items():
            if key in prompt.user:
                return reply
        return "{}"

    return StaticLanguageModel(_reply)


# ---------------- metric functions ----------------


def test_content_hit():
    assert content_hit("Apply a COATING now", ["coating"]) is True
    assert content_hit("apply paint", ["coating"]) is False


def test_token_f1():
    assert token_f1("apply a coating", "apply a coating") == 1.0
    assert token_f1("apply coating", "apply a coating") == 1.0  # the article is normalized away
    assert token_f1("the red car", "a red truck") == 0.5  # one shared token of two each
    assert token_f1("zebra", "coating") == 0.0


def test_exact_match():
    assert exact_match("Apply a coating.", "apply a coating") is True  # punctuation + article normalized
    assert exact_match("apply coatings", "apply a coating") is False


def test_citation_coverage_counts_only_cited_passages():
    result = GenerationResult(
        answer="x",
        proof=[ProofStep(claim="c", citations=[Citation(chunk_id="a", document_id="d")])],
        evidence=[_result("a", "Coatings resist corrosion."), _result("b", "Quokkas grin.")],
    )
    assert citation_coverage(result, ["corrosion"]) == 1.0
    assert citation_coverage(result, ["zebra"]) == 0.0
    assert citation_coverage(result, ["corrosion", "zebra"]) == 0.5
    assert citation_coverage(result, ["quokkas"]) == 0.0  # "b" is in evidence but not cited -> doesn't count


# ---------------- end-to-end ----------------


async def test_evaluate_generation_scores_all_four_axes():
    evidence = [_result("a", "Coatings resist corrosion.")]
    llm = _routed_lm(
        {
            "rust": json.dumps(
                {"answer": "apply a coating", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
            ),
            "france": json.dumps(
                {"answer": "Paris", "steps": [{"claim": "the capital is paris", "cited": [1]}]}
            ),
        }
    )
    ctx = GenerationContext(_Retrieval(evidence), llm)
    evalset = GenEvalSet(
        [
            GenEvalQuery(
                text="how to prevent rust?",
                answer_contains=["coating"],
                answer="apply a coating",
                supporting=["corrosion"],
            ),
            GenEvalQuery(text="capital of france?", should_abstain=True),  # ungrounded here -> should abstain
        ]
    )
    # grounding + abstain so the unanswerable query refuses (its claim doesn't match the coatings passage)
    spec = {
        "class_name": "generation_pipeline",
        "grounding_checker": {"class_name": "heuristic_grounding"},
        "abstain": True,
    }
    pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
    report = await evaluate_generation(pipeline, ctx, evalset)

    assert report.n == 2
    assert report.content_hit == 1.0 and report.token_f1 == 1.0 and report.exact_match == 1.0
    assert report.grounded_rate == 1.0  # the one non-abstained answer was grounded
    assert report.abstention_accuracy == 1.0  # answered the answerable, abstained the unanswerable
    assert report.citation_coverage == 1.0  # the cited passage covers "corrosion"

    q1, q2 = report.per_query
    assert q1.abstained is False and q1.content_hit is True
    assert q2.abstained is True and q2.abstention_correct is True and q2.content_hit is None  # not applicable


async def test_sweep_generation_and_format():
    ctx = GenerationContext(
        _Retrieval([_result("a", "Coatings resist corrosion.")]),
        StaticLanguageModel(
            json.dumps({"answer": "apply a coating", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]})
        ),
    )
    evalset = GenEvalSet([GenEvalQuery(text="prevent rust?", answer_contains=["coating"], answer="apply a coating")])
    reports = await sweep_generation({"single": {"class_name": "generation_pipeline"}}, ctx, evalset)
    assert set(reports) == {"single"} and reports["single"].content_hit == 1.0
    table = format_generation_reports(reports)
    assert "single" in table and "hit" in table and "cite" in table
