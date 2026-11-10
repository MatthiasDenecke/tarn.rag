"""CRIT-01: generation retrieves FOR grounding — so a chunk ingested with ``ai_grounding_allowed=False``
can be *retrieved* (EXECUTION) but never reaches an LLM prompt. Grounding is additive: the caller's
purpose is kept, so generation's license classes are its purpose's intersected with
``GENERATION_GROUNDING``'s — tighter, never wider (a narrowed map must not be widened by ``ask``).

Unit-level (the pipeline is the one choke point, and the reasoners carry the flag into every sub-query)
plus end-to-end runs through the facade on the model-free ``hash`` embedder."""

import json

from bausatz import ComponentFactory

from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import GenerationContext, GenerationPipeline
from tarnrag.retrieval.types import Purpose, Query
from tarnrag.tarnrag import TarnRag


class _RecordingRetrieval:
    """A retrieval-port double recording every query it is asked (the reasoners may ask several)."""

    def __init__(self) -> None:
        self.queries: list[Query] = []

    async def search(self, query: Query) -> list:
        self.queries.append(query)
        return []


async def test_the_pipeline_adds_grounding_and_keeps_the_callers_purpose():
    port = _RecordingRetrieval()
    pipeline = ComponentFactory.get().create_as(
        {"class_name": "generation_pipeline", "reasoner": {"class_name": "decomposition"}}, GenerationPipeline
    )
    decompose = json.dumps({"sub_questions": ["who built it?", "when?"]})
    llm = StaticLanguageModel(lambda p: decompose if "sub_questions" in p.system else "{}")
    caller_query = Query(text="who built the tower and when?", purpose=Purpose.EXECUTION)
    await pipeline.answer(caller_query, GenerationContext(port, llm))
    assert len(port.queries) == 2  # one search per sub-question
    assert all(q.grounds_generation for q in port.queries)  # every sub-query is retrieved for grounding
    assert {q.purpose for q in port.queries} == {Purpose.EXECUTION}  # …under the caller's own purpose
    assert all(q.permitted_filter().require_grounding for q in port.queries)
    assert not caller_query.grounds_generation  # the caller's query is not mutated


def _policy(permitted):
    return ComponentFactory.get().create({"class_name": "default_license", "permitted": permitted})


def test_grounding_intersects_the_purposes_license_classes_never_widens_them():
    """The review repro: a map that narrows EXECUTION. Generation (EXECUTION + grounding) must stay within
    EXECUTION's classes — the old purpose swap handed it GENERATION_GROUNDING's wider set instead."""
    policy = _policy({
        "EXECUTION": ["customer_licensed"],
        "GENERATION_GROUNDING": ["customer_licensed", "public_domain", "modusq_authored"],
    })
    grounded = policy.filter_for(Query(text="q", purpose=Purpose.EXECUTION, grounds_generation=True))
    assert grounded.license_classes == ("customer_licensed",) and grounded.require_grounding
    plain = policy.filter_for(Query(text="q", purpose=Purpose.EXECUTION))
    assert plain.license_classes == ("customer_licensed",) and not plain.require_grounding
    # a deployment that narrows GENERATION_GROUNDING instead is honored too
    narrow_gg = _policy({"EXECUTION": ["customer_licensed", "public_domain"], "GENERATION_GROUNDING": ["public_domain"]})
    assert narrow_gg.filter_for(
        Query(text="q", grounds_generation=True)
    ).license_classes == ("public_domain",)


def test_a_purpose_absent_from_the_map_fails_closed():
    """An absent purpose permits nothing — it used to mean "no license filter at all", which let
    ``third_party_copyrighted`` through for any purpose a custom map forgot."""
    policy = _policy({"EXECUTION": ["customer_licensed"]})
    assert policy.filter_for(Query(text="q", purpose=Purpose.AUTHORING)).license_classes == ()
    assert policy.filter_for(Query(text="q", grounds_generation=True)).license_classes == ()  # GG absent


def _hash_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        MODE="embedded",
        ID_POLICY="caller",
        EMBEDDING_DIMENSION=64,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/grounding.db"),
        components={"generation_pipeline": {"class_name": "generation_pipeline"}},  # single_hop reader
    )


async def test_a_grounding_disallowed_document_is_retrievable_but_never_grounds_an_answer(tmp_path):
    prompts: list[str] = []

    def _reply(prompt) -> str:
        prompts.append(prompt.user)
        return json.dumps({"answer": "ok", "steps": []})

    async with TarnRag(_hash_settings(tmp_path), llm=StaticLanguageModel(_reply)) as tarn:
        ingestion = await tarn._ingestion_engine()
        await ingestion.ingest_content([
            {"source_id": "open", "content": "acme widgets ship from the north depot"},
            {"source_id": "licensed", "content": "acme widgets ship from the south depot",
             "ai_grounding_allowed": 0},
        ])

        retrieved = await tarn.retrieve("acme widgets depot")
        assert {r.document_id for r in retrieved.value} == {"open", "licensed"}  # EXECUTION sees both

        answer = await tarn.ask("acme widgets depot")
        assert {r.document_id for r in answer.value.evidence} == {"open"}
        assert prompts and all("south depot" not in p for p in prompts)  # never laid before the reader

        # The engine path with an explicit (non-grounding) Query is protected the same way.
        generated = await tarn._gen().answer(Query(text="acme widgets depot", purpose=Purpose.EXECUTION))
        assert {r.document_id for r in generated.evidence} == {"open"}


async def test_ask_never_sees_a_license_class_the_callers_purpose_excludes(tmp_path):
    """End to end: EXECUTION narrowed to customer_licensed; ``ask`` (EXECUTION by default) must not reach
    the public_domain document even though GENERATION_GROUNDING would permit it."""
    settings = _hash_settings(tmp_path)
    settings.components["license_policy"] = {
        "class_name": "default_license",
        "permitted": {
            "EXECUTION": ["customer_licensed"],
            "GENERATION_GROUNDING": ["customer_licensed", "public_domain"],
        },
    }
    async with TarnRag(settings, llm=StaticLanguageModel(json.dumps({"answer": "ok", "steps": []}))) as tarn:
        ingestion = await tarn._ingestion_engine()
        await ingestion.ingest_content([
            {"source_id": "ours", "content": "acme widgets ship weekly", "license_class": "customer_licensed"},
            {"source_id": "public", "content": "acme widgets ship daily", "license_class": "public_domain"},
        ])
        retrieved = await tarn.retrieve("acme widgets ship")
        assert {r.document_id for r in retrieved.value} == {"ours"}
        answer = await tarn.ask("acme widgets ship")
        assert {r.document_id for r in answer.value.evidence} == {"ours"}  # not widened to public_domain
