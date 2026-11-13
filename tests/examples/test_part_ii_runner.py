"""The Part II example runner (``examples.part_ii._runner``) — model-free, via the ``hash`` embedder.

These exercise the runner's plumbing (ingest → explain → render → verdict) without the ONNX model: the
``hash`` embedder is deterministic but has **no semantics**, so we never assert *which* document ranks
where — only that the runner runs end to end and that its HIT/MISS verdict matches the actual retrieval.
"""

from __future__ import annotations

from examples.common import corpus, load_config
from examples.part_ii._runner import Runner
from tarnrag import TarnRag
from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings


def test_example_00_base_config_is_explicit_and_complete():
    """The base config fully specifies all three pipelines + the OpenAI reader (no model needed to load)."""
    s = load_config("examples/part_ii/example_00/config.yaml")
    assert s.components["ingestion_pipeline"]["stages"][0]["routes"] == {"md": {"class_name": "markdown"}}
    assert s.components["retrieval_pipeline"]["retrievers"][0]["class_name"] == "dense"
    assert s.components["generation_pipeline"]["reasoner"]["class_name"] == "single_hop"
    assert s.llm.provider == "openai" and s.llm.key_env_var() == "OPENAI_LLM_KEY"


def _hash_settings(tmp_path) -> Settings:
    """Embedded settings over a temp store + the model-free ``hash`` embedder (no model dir needed)."""
    return Settings(
        _env_file=None, MODE="embedded", EMBEDDING_DIMENSION=64, ID_POLICY="caller",
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/runner.db"),
        embedding=EmbeddingSettings(provider="hash"),
    )


async def test_runner_ingests_corpus_and_renders_without_error(tmp_path):
    async with TarnRag(_hash_settings(tmp_path)) as tarn:
        runner = Runner(tarn)
        runner.banner("P2-test", shows="the runner", fails="nothing", fixed_next="n/a")
        await runner.ingest_corpus(corpus("corpus-2"))
        docs = (await tarn.docs()).value
        assert len(docs) == 12  # the corpus-2 markdown files all ingested


async def test_runner_retrieval_verdict_matches_actual_retrieval(tmp_path):
    """``show_retrieval``'s HIT/MISS must agree with a direct retrieval check — content-based, label-free,
    and independent of the (meaningless) hash ranking."""
    async with TarnRag(_hash_settings(tmp_path)) as tarn:
        runner = Runner(tarn)
        await runner.ingest_corpus(corpus("corpus-2"))
        results = (await tarn.retrieve("storage tank inspection", top_k=8)).value
        expected = any("quokka" in r.text.lower() for r in results)

        verdict = await runner.show_retrieval(
            "storage tank inspection", gold="quokka", top_k=8, label="probe"
        )
        assert verdict is expected
        assert await runner.show_retrieval("anything", gold=None) is None  # no gold -> no verdict
