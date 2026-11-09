"""CRIT-01: generation retrieves under ``Purpose.GENERATION_GROUNDING`` — so a chunk ingested with
``ai_grounding_allowed=False`` can be *retrieved* (EXECUTION) but never reaches an LLM prompt.

Unit-level (the pipeline is the one choke point, and the reasoners carry the purpose into every
sub-query) plus an end-to-end run through the facade on the model-free ``hash`` embedder."""

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


async def test_the_pipeline_retrieves_for_grounding_whatever_the_callers_purpose():
    port = _RecordingRetrieval()
    pipeline = ComponentFactory.get().create_as(
        {"class_name": "generation_pipeline", "reasoner": {"class_name": "decomposition"}}, GenerationPipeline
    )
    decompose = json.dumps({"sub_questions": ["who built it?", "when?"]})
    llm = StaticLanguageModel(lambda p: decompose if "sub_questions" in p.system else "{}")
    caller_query = Query(text="who built the tower and when?", purpose=Purpose.EXECUTION)
    await pipeline.answer(caller_query, GenerationContext(port, llm))
    assert len(port.queries) == 2  # one search per sub-question
    assert {q.purpose for q in port.queries} == {Purpose.GENERATION_GROUNDING}
    assert caller_query.purpose is Purpose.EXECUTION  # the caller's query is not mutated


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
