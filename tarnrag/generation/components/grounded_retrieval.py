"""GroundedRetrievalReasoner — multi-hop by γ-driven re-retrieval (MOTHRAG §3.4).

Unlike ``IterativeReasoner`` (which loops on the model's self-reported ``done`` / ``follow_up``), this loop
is driven by the **grounding check**: retrieve → read → check each claim; a claim that isn't ``GROUNDED``
triggers a follow-up search **for that claim**, pooling evidence the question alone couldn't surface (the
multi-hop *bridge* passage — similar to the first hop's answer, not to the question), then re-read — until
every claim is grounded or ``max_hops`` is reached.

It lives in its own module (not ``reasoner.py``) because it imports both the ``Reasoner`` base and the
``GroundingChecker`` — and ``grounding`` already imports ``reasoner`` (``ReasonedAnswer``), so importing
``grounding`` from ``reasoner`` would be a cycle. Registered by ``class_name`` when the module is imported
(see ``tarnrag.generation.__init__``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import Field

from tarnrag.contracts import RetrievalResult
from bausatz import ComponentFactory
from tarnrag.core.resources.llm import Prompt
from tarnrag.generation.components.grounding import (
    GroundingChecker,
    HeuristicGroundingChecker,
    Verdict,
)
from tarnrag.generation.components.reasoner import _READ_SYSTEM, ReasonedAnswer, Reasoner
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


class GroundedRetrievalReasoner(Reasoner):
    """Multi-hop by grounding-driven re-retrieval: an ungrounded claim triggers a follow-up search *for that
    claim*, finding the bridge evidence dense retrieval on the question couldn't, then re-read — until every
    claim is grounded or ``max_hops``. ``grounding_checker`` is the γ check (default the LLM-free heuristic;
    swap to ``llm_grounding`` / ``cascading_grounding`` for a sharper, costlier signal)."""

    class Config(Reasoner.Config):
        class_name: Literal["grounded_retrieval"] = "grounded_retrieval"
        grounding_checker: GroundingChecker.Config = Field(
            default_factory=HeuristicGroundingChecker.Config
        )
        max_hops: int = 3
        top_k: int = 8  # passages retrieved for the question and for each follow-up claim

    config: GroundedRetrievalReasoner.Config

    def __init__(self, config: GroundedRetrievalReasoner.Config) -> None:
        super().__init__(config)
        self._grounding: GroundingChecker | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the grounding-checker child (the γ signal that drives re-retrieval)."""
        self._grounding = factory.build_as(self.config.grounding_checker, GroundingChecker)

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        self.ensure_built()
        evidence: list[RetrievalResult] = []
        seen: set[str] = set()
        self._accumulate(evidence, seen, await ctx.retrieval.search(replace(query, top_k=self.config.top_k)))
        reasoned = ReasonedAnswer(answer="", steps=[], evidence=evidence)
        for hop in range(self.config.max_hops):
            user = f"Question: {query.text}\n\nPassages:\n{self._numbered_passages(evidence)}"
            completion = await ctx.llm.complete(Prompt(system=_READ_SYSTEM, user=user))
            answer, steps = self._read(completion.text, len(evidence))
            reasoned = ReasonedAnswer(answer=answer, steps=steps, evidence=evidence)
            if hop == self.config.max_hops - 1:
                break
            verdicts = await self._grounding.check(reasoned, ctx)
            gaps = [s.claim for s, v in zip(steps, verdicts) if v is not Verdict.GROUNDED]
            if not gaps:
                break  # every claim grounded — done
            before = len(evidence)
            for claim in gaps:  # re-retrieve for the unsupported claim(s) — the bridge-passage hop
                self._accumulate(
                    evidence, seen, await ctx.retrieval.search(replace(query, text=claim, top_k=self.config.top_k))
                )
            if len(evidence) == before:
                break  # the follow-up found nothing new — another read won't help
        return reasoned
