"""The answerability gate (PP-2) — refuse *before* the read when the evidence can't support the query.

A wrapper ``Reasoner``: it probes retrieval once and checks that the query's **exact-match cues** —
double-quoted spans and identifier tokens (``§6.4.2``, error codes, part numbers; the same
``core.text`` cues the classifier labels and the sparse builder phrases) — actually appear in the
retrieved evidence. A cue no passage covers means the corpus doesn't mention the thing being asked
about, so the gate returns a refusal naming what's missing instead of letting the reader guess
(``abstained=True`` — the pipeline surfaces it without spending the read or the verification).
Queries with no checkable cues pass straight through to the wrapped reasoner, as do covered ones
(the child re-retrieves; retrieval is cheap — the LLM read is the cost being saved). Coverage is
checked against the reader-facing rendering (``passage_text``), so a value inside a table counts.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from bausatz import ComponentFactory
from pydantic import Field

from tarnrag.core.text import looks_like_identifier, quoted_spans
from tarnrag.generation.components._passages import passage_text
from tarnrag.generation.components.reasoner import (
    ReasonedAnswer,
    Reasoner,
    SingleHopReasoner,
)
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


class AnswerabilityGateReasoner(Reasoner):
    """Gate a wrapped reasoner on evidence coverage of the query's exact-match cues (see module doc)."""

    class Config(Reasoner.Config):
        class_name: Literal["answerability"] = "answerability"
        reasoner: Reasoner.Config = Field(default_factory=SingleHopReasoner.Config)
        top_k: int = 8  # the gate's evidence probe
        refusal: str = (
            "The available evidence does not mention {missing}, so I can't answer that reliably."
        )

    config: AnswerabilityGateReasoner.Config

    def __init__(self, config: AnswerabilityGateReasoner.Config) -> None:
        super().__init__(config)
        self._child: Reasoner | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        self._child = factory.build_as(self.config.reasoner, Reasoner)

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        self.ensure_built()
        cues = self._cues(query.text)
        if not cues:
            return await self._child.reason(query, ctx)  # nothing checkable — pass through
        results = await ctx.retrieval.search(replace(query, top_k=self.config.top_k))
        haystacks = [self._normalize(passage_text(r, self.config.table_view)) for r in results]
        missing = [cue for cue in cues if not self._covered(cue, haystacks)]
        if missing:
            return ReasonedAnswer(
                answer=self.config.refusal.format(missing=", ".join(missing)),
                steps=[],
                evidence=results,  # what WAS found — a UI can show why the gate refused
                abstained=True,
            )
        return await self._child.reason(query, ctx)

    @staticmethod
    def _cues(text: str) -> list[str]:
        """The checkable exact-match cues, deduped in order: quoted spans, then identifier tokens."""
        cues: list[str] = []
        for cue in quoted_spans(text) + [t for t in text.split() if looks_like_identifier(t)]:
            if cue not in cues:
                cues.append(cue)
        return cues

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.lower()))

    @classmethod
    def _covered(cls, cue: str, haystacks: list[str]) -> bool:
        """Whether the cue's normalized token run appears (adjacent, in order) in any passage. A cue
        that normalizes to nothing (pure punctuation) is vacuously covered."""
        needle = cls._normalize(cue)
        return not needle or any(needle in haystack for haystack in haystacks)
