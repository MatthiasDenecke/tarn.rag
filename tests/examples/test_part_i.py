"""Smoke tests for the Part I examples.

Each test runs the example's own ``main()`` against a temp store (via ``EXAMPLES_DATA_DIR``) and
checks it produces sensible output — so the suite catches API drift in the examples before an
onboarding developer does. Gated on ``requires_model`` (auto-skipped when the model/runtime
aren't present; see ``conftest.py``).

The pattern every Part I example slots into: import its module(s), run ``main()`` under the
isolated store, assert on the returned data.
"""

from __future__ import annotations

import json

import pytest

from examples.part_i.example_01 import ingestion, retrieval
from examples.part_i.example_02 import ingestion as ingestion_02
from examples.part_i.example_02 import retrieval as retrieval_02
from examples.part_i.example_03 import evaluation as evaluation_03
from examples.part_i.example_03 import ingestion as ingestion_03
from examples.part_i.example_04 import generation as generation_04
from examples.part_i.example_04 import ingestion as ingestion_04
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.eval import by_query_type

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Redirect each example's SQLite store into a temp dir, so tests never write into the tree."""
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_01_ingest_then_retrieve():
    # Ingestion: the corpus is fully indexed (one chunk + embedding per short doc).
    statuses = await ingestion.main()
    assert len(statuses) == 3
    assert all(s.status == "complete" for s in statuses)
    assert all(s.chunk_count >= 1 and s.embedding_count >= 1 for s in statuses)

    # Retrieval over the same store ranks sensibly: the corrosion query surfaces the tank doc.
    answers = await retrieval.main()
    corrosion_query = retrieval.QUERIES[0]
    assert answers[corrosion_query][0].document_id == "tank-inspection"


async def test_example_02_json_pipeline_makes_more_chunks():
    # pipeline.json uses small chunks, so the same corpus splits into more chunks than example 01's
    # default (one chunk per short doc) — proof the JSON actually drove the pipeline composition.
    statuses = await ingestion_02.main()
    assert len(statuses) == 3
    assert all(s.status == "complete" for s in statuses)
    assert sum(s.chunk_count for s in statuses) > 3  # more than the default's one-per-doc

    # Retrieval over the small-chunk store still ranks the corrosion query onto the tank doc.
    answers = await retrieval_02.main()
    assert answers[retrieval_02.QUERIES[0]][0].document_id == "tank-inspection"


async def test_example_03_compares_retrieval_methods():
    # Ingestion: the small-chunk pipeline gives several chunks per doc for the methods to differ over.
    statuses = await ingestion_03.main()
    assert len(statuses) == 3 and all(s.status == "complete" for s in statuses)
    assert sum(s.chunk_count for s in statuses) > 3

    # The sweep scores every configured pipeline spec — the three methods + the router — over the
    # labeled (semantic vs lexical) set.
    reports = await evaluation_03.main()
    assert set(reports) == {"dense", "sparse (bm25)", "hybrid (rrf)", "routed (by type)"}
    assert all(r.n == 6 for r in reports.values())

    # The segmentation surfaces what the aggregate hides: sparse BM25 collapses on the semantic
    # (paraphrase) queries — no shared terms — while it ties on the lexical ones; dense is robust on both.
    sparse = by_query_type(reports["sparse (bm25)"])
    dense = by_query_type(reports["dense"])
    routed = by_query_type(reports["routed (by type)"])
    assert set(sparse) == {"lexical", "semantic"}
    assert sparse["semantic"].hit_at_k < sparse["lexical"].hit_at_k       # the collapse
    assert dense["semantic"].hit_at_k >= sparse["semantic"].hit_at_k      # dense more robust on semantic

    # The router sends each type to its configured method, so per type it matches that method — taking
    # sparse's lexical strength and dense's semantic robustness instead of either single method's weak side.
    assert routed["lexical"].hit_at_k == sparse["lexical"].hit_at_k
    assert routed["semantic"].hit_at_k == dense["semantic"].hit_at_k


# Canned answers for example 04, routed on the question line (not the whole prompt, whose passages would
# false-match). Lets the smoke test exercise the real reasoner + grounding + abstention flow with no key.
_EX04_ANSWERS = {
    "pump": {"answer": "Check the mechanical seal, bearing lubrication, and shaft alignment.",
             "steps": [{"claim": "service the mechanical seal and bearing lubrication", "cited": [1]}]},
    "corrosion": {"answer": "Visual inspection plus ultrasonic shell-thickness testing.",
                  "steps": [{"claim": "ultrasonic testing measures shell thickness for corrosion", "cited": [1]}]},
    "capital": {"answer": "Paris.", "steps": [{"claim": "the capital of france is paris", "cited": [1]}]},
}


def _example04_llm() -> StaticLanguageModel:
    def _reply(prompt):
        if prompt.system and "fact-checker" in prompt.system:  # the grounding checker's prompt
            return json.dumps({"verdicts": [True, True, True, True]})  # say grounded (parser pads to n)
        question_line = prompt.user.split("\n", 1)[0].lower()  # "question: ..."
        for key, reply in _EX04_ANSWERS.items():
            if key in question_line:
                return json.dumps(reply)
        return json.dumps({"answer": "I don't know.", "steps": []})

    return StaticLanguageModel(_reply)


async def test_example_04_generation_end_to_end():
    # Ingestion is offline (local embedder); only generation needs an LLM, which we inject (no key).
    statuses = await ingestion_04.main()
    assert len(statuses) == 3 and all(s.status == "complete" for s in statuses)

    results = await generation_04.main(llm=_example04_llm())
    assert len(results) == 3  # one GenerationResult per question in questions.json

    # The two corpus questions are answered + grounded (their claims overlap the cited passages);
    # the off-topic "capital of France" has no support in the corpus -> the policy abstains.
    pump, corrosion, france = results
    assert not pump.abstained and pump.grounded and pump.proof[0].grounded
    assert not corrosion.abstained and corrosion.grounded
    assert france.abstained  # the abstention policy refused the unsupported question
    # The proof tree cites the readable source document the answer rests on.
    assert pump.proof[0].citations[0].document_id == "pump-maintenance"
