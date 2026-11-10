"""The GenerationPipeline + GenerationEngine: reason -> assemble proof tree -> GenerationResult."""

import json

from bausatz import ComponentFactory

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import GenerationContext, GenerationEngine, GenerationPipeline
from tarnrag.retrieval.types import Query


def _pipeline(spec=None) -> GenerationPipeline:
    return ComponentFactory.get().create_as(spec or {"class_name": "generation_pipeline"}, GenerationPipeline)


async def test_pipeline_builds_proof_tree_with_provenance(make_result, fake_retrieval):
    results = [
        make_result("a", "Coatings resist corrosion.", header_path=["Maint", "Corrosion"], start=0, end=24),
        make_result("b", "Quokkas grin."),
    ]
    reply = json.dumps(
        {"answer": "Apply a coating.", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(reply))
    res = await _pipeline().answer(Query(text="prevent rust?"), ctx)

    assert res.answer == "Apply a coating."
    assert res.grounded is True and res.abstained is False  # slice-2 defaults (grounding is slice 3)
    assert [r.chunk_id for r in res.evidence] == ["a", "b"]

    [step] = res.proof
    [cite] = step.citations
    assert cite.chunk_id == "a"
    assert cite.header_path == ["Maint", "Corrosion"]
    assert cite.geometry[0].start == 0 and cite.geometry[0].end == 24  # highlightable span from provenance


async def test_engine_answer_text_delegates(make_result, fake_retrieval):
    results = [make_result("a", "x")]
    llm = StaticLanguageModel(json.dumps({"answer": "ok"}))
    eng = GenerationEngine(fake_retrieval(results), llm, _pipeline())
    res = await eng.answer_text("q")
    assert res.answer == "ok" and len(res.evidence) == 1


async def test_engine_is_an_async_context_manager(make_result, fake_retrieval):
    # A directly-constructed engine runs over INJECTED resources (owns_resources=False), so __aexit__ ->
    # aclose closes nothing — their owner does. Must be a clean no-op, not an error.
    async with GenerationEngine(fake_retrieval([]), StaticLanguageModel("{}"), _pipeline()) as eng:
        assert eng.retrieval is not None


class _ClosablePort:
    """A retrieval-port double that records whether ``aclose`` was called."""

    def __init__(self):
        self.closed = False

    async def search(self, query):
        return []

    async def aclose(self):
        self.closed = True


class _ClosableLLM(StaticLanguageModel):
    def __init__(self):
        super().__init__("{}")
        self.closed = False

    async def aclose(self):
        self.closed = True


async def test_aclose_releases_owned_resources():
    # The standalone-create posture: the engine built its retrieval + LLM, so aclose releases both.
    port, llm = _ClosablePort(), _ClosableLLM()
    await GenerationEngine(port, llm, _pipeline(), owns_resources=True).aclose()
    assert port.closed and llm.closed


async def test_aclose_leaves_injected_resources_to_their_owner():
    # The composition-root posture (TarnRag): retrieval + LLM are injected and shared — never closed here.
    port, llm = _ClosablePort(), _ClosableLLM()
    await GenerationEngine(port, llm, _pipeline()).aclose()
    assert not port.closed and not llm.closed


async def test_create_builds_and_owns_its_resources(monkeypatch):
    """The standalone ``create`` path builds the retrieval engine + LLM itself (factories stubbed here),
    so the engine owns them — ``aclose`` releases both."""
    from tarnrag.core.engine.config import Settings
    from tarnrag.core.resources.llm import LanguageModel
    from tarnrag.retrieval.engine.engine import RetrievalEngine

    port, llm = _ClosablePort(), _ClosableLLM()

    async def fake_retrieval_create(settings):
        return port

    monkeypatch.setattr(RetrievalEngine, "create", fake_retrieval_create)
    monkeypatch.setattr(LanguageModel, "create", lambda settings: llm)
    eng = await GenerationEngine.create(Settings(_env_file=None))
    assert eng.retrieval is port and eng.llm is llm
    await eng.aclose()
    assert port.closed and llm.closed  # created standalone ⇒ owned ⇒ released


async def test_no_results_yields_an_answer_with_empty_evidence(make_result, fake_retrieval):
    # The model is told there are no passages; the pipeline still returns a (cite-nothing) result.
    ctx = GenerationContext(fake_retrieval([]), StaticLanguageModel(json.dumps({"answer": "No passages found."})))
    res = await _pipeline().answer(Query(text="q"), ctx)
    assert res.answer == "No passages found." and res.evidence == []
    assert res.proof and res.proof[0].citations == []  # one step, nothing to cite


def _grounded(**extra):
    """A generation spec with the heuristic grounding checker (no LLM, so it doesn't clash with the
    reasoner's canned reply) + any abstention overrides."""
    return {"class_name": "generation_pipeline", "grounding_checker": {"class_name": "heuristic_grounding"}, **extra}


async def test_grounding_stamps_verdicts_and_keeps_a_supported_answer(make_result, fake_retrieval):
    results = [make_result("a", "Coatings resist corrosion."), make_result("b", "Quokkas grin.")]
    reply = json.dumps(
        {"answer": "Apply a coating.", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(reply))
    res = await _pipeline(_grounded()).answer(Query(text="?"), ctx)
    assert res.proof[0].grounded is True
    assert res.grounded is True and res.abstained is False


async def test_grounding_flags_an_unsupported_answer_without_abstaining(make_result, fake_retrieval):
    results = [make_result("a", "Coatings resist corrosion."), make_result("b", "Quokkas grin.")]
    # the claim cites passage 1 (coatings) but is about quokkas -> ungrounded
    reply = json.dumps(
        {"answer": "Quokkas are great.", "steps": [{"claim": "quokkas are marsupials", "cited": [1]}]}
    )
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(reply))
    res = await _pipeline(_grounded()).answer(Query(text="?"), ctx)
    assert res.proof[0].grounded is False
    assert res.grounded is False and res.abstained is False  # best-grounded flag (abstain off by default)
    assert res.answer == "Quokkas are great."  # the answer is kept


async def test_grounding_abstains_below_threshold(make_result, fake_retrieval):
    results = [make_result("a", "Coatings resist corrosion.")]
    reply = json.dumps(
        {"answer": "Quokkas are great.", "steps": [{"claim": "quokkas are marsupials", "cited": [0]}]}
    )
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(reply))
    res = await _pipeline(_grounded(abstain=True)).answer(Query(text="?"), ctx)
    assert res.abstained is True and res.grounded is False
    assert "grounded evidence" in res.answer.lower()  # the refusal replaced the answer


async def test_no_checker_preserves_grounded_default(make_result, fake_retrieval):
    results = [make_result("a", "x")]
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(json.dumps({"answer": "ok"})))
    res = await _pipeline().answer(Query(text="?"), ctx)
    assert res.grounded is True and res.abstained is False and res.proof[0].grounded is True
