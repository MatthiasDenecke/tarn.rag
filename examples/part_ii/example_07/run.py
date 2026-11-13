"""Part II · Example 07 — grounding check + abstain.

Example 06 *trusted* the reader. Here the generation_pipeline gains two things:
  * a `grounding_checker` re-reads each claim against its cited evidence and stamps it grounded/ungrounded;
  * an `abstain` policy refuses the answer outright when fewer than `min_grounded` of the claims hold up.

Two parts, because the two halves have different reproducibility:
  * LIVE (console-runnable) — the good answer is now VERIFIED: llm_grounding re-reads the claim against the
    evidence, the proof's ✓ is earned (not assumed), and the question is still answered.
  * CONSTRUCTED (deterministic) — to SHOW the abstain firing we must plant an unsupported claim, because the
    honest live reader (gpt-4o) doesn't fabricate: it answers when supported and says "not provided"
    otherwise. So, exactly as the library's own grounding tests do, we feed a fixed reply through a
    grounding+abstain pipeline (heuristic_grounding — content-word overlap, no LLM) and watch it refuse.

Run from the repo root::

    python -m examples.part_ii.example_07.run

The live half needs the LLM key in OPENAI_LLM_KEY (repo-root .env) + `pip install '.[openai]'`; the
constructed half needs neither.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bausatz import ComponentFactory

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag
from tarnrag.contracts import ChunkProvenance, RetrievalResult
from tarnrag.contracts.structure import Span
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import GenerationContext, GenerationPipeline, GenerationResult
from tarnrag.retrieval import Query

CONFIG = Path(__file__).resolve().parent / "config.yaml"


class _FakeRetrieval:
    """A retrieval-port double returning canned evidence — so the constructed scenario needs no store."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def search(self, query: Query) -> list[RetrievalResult]:
        return list(self._results)


def _planted_scenario() -> GenerationContext:
    """One real-looking evidence chunk (pump maintenance — which says nothing about a warranty) paired with
    a reply the reader did NOT ground: a fabricated ten-year-warranty claim citing that maintenance passage."""
    text = "Before restarting a centrifugal pump after maintenance, check the mechanical seal for leaks."
    evidence = [
        RetrievalResult(
            chunk_id="pump-maintenance#0", text=text, score=1.0, component_scores={},
            document_id="pump-maintenance", source_kind="markdown", standard_id=None, locator=None,
            license_class="open",
            provenance=ChunkProvenance(
                geometry=[Span(start=0, end=len(text))],
                header_path=["Centrifugal pump maintenance"], content_hash="h",
            ),
        )
    ]
    reply = json.dumps({
        "answer": "The pump is covered by a ten-year manufacturer warranty.",
        "steps": [{"claim": "the manufacturer warranty lasts a decade", "cited": [1]}],
    })
    return GenerationContext(_FakeRetrieval(evidence), StaticLanguageModel(reply))


async def planted_abstain() -> GenerationResult:
    """Run the planted (ungrounded) reply through a grounding+abstain pipeline. DETERMINISTIC — no live LLM
    and no store: heuristic_grounding (content-word overlap) flags the disjoint warranty claim, and the
    abstain policy returns the refusal. Mirrors the library's own grounding tests."""
    spec = {
        "class_name": "generation_pipeline",
        "grounding_checker": {"class_name": "heuristic_grounding"},  # no LLM → deterministic
        "abstain": True, "min_grounded": 1.0,
    }
    pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
    return await pipeline.answer(Query(text="how long is the pump under warranty?"), _planted_scenario())


async def main() -> tuple[bool, bool]:
    """Return (the good answer was verified-grounded and still answered, the planted claim was refused)."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 07 · grounding check + abstain",
            shows="a grounding_checker verifies each claim against its cited evidence (the proof's ✓ is now "
            "earned, not assumed); the abstain policy refuses an unsupported answer",
            fixed_next="Example 08 — multi-hop: decompose a question whose answer spans two documents",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        # LIVE: the answerable question, now VERIFIED — llm_grounding re-read the claim, then it answered.
        good = await runner.show_answer(
            "How do I service a centrifugal pump before starting it?",
            gold="mechanical seal", label="verified · the claim was checked against its evidence, then answered",
        )
        # CONSTRUCTED: the honest live reader won't fabricate, so (as the library's tests do) we plant an
        # unsupported claim and watch the grounding check refuse it — deterministically, no live LLM.
        runner.note("constructed illustration — a planted unsupported claim (the live reader won't fabricate one):")
        refused = runner.show_result(
            await planted_abstain(),
            label="abstained · a fabricated warranty claim, caught by the grounding check",
        )
        good_ok = good.grounded and not good.abstained and "mechanical seal" in good.answer.lower()
        return good_ok, refused.abstained


if __name__ == "__main__":
    asyncio.run(main())
