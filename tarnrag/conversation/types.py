"""Conversation result types — what contextualizing one question produced.

A ``Contextualization`` pairs the question **as typed** with the question **as executed** (the
standalone rewrite), plus how it came about: how many remembered turns were consulted, which
contextualizer did it, and — when a follow-up could *not* be resolved — why the question was used as
typed after all. It is plain text in, plain text out: the conversation package never builds a
``Query`` (it imports neither ``retrieval`` nor ``generation``); the facade does, attaching
:meth:`Contextualization.annotations` so a rewrite travels with the query, flagged as generated.
"""

from __future__ import annotations

from dataclasses import dataclass

from tarnrag.contracts import Annotation

# The ``Annotation.type`` a contextualized query carries (its ``value`` holds the original wording).
CONTEXTUALIZED = "contextualized"


@dataclass(frozen=True)
class Contextualization:
    """One question, before and after the conversation was taken into account."""

    original: str  # the question as typed
    standalone: str  # the question as executed (== original when nothing was rewritten)
    turns_used: int = 0  # how many remembered turns the contextualizer was shown
    producer: str = ""  # the contextualizer's class_name ("" when none ran: memory off / empty window)
    fallback: str | None = None  # why the question was used as typed although history existed

    @property
    def rewritten(self) -> bool:
        """Whether the executed question differs from the typed one."""
        return self.standalone != self.original

    def annotations(self) -> list[Annotation]:
        """The annotation a query built from this carries — one ``contextualized`` finding recording the
        original wording, flagged ``deterministic=False`` (an LLM produced it; never silently trusted).
        Empty when no contextualizer ran or it fell back: the query is then simply the question as typed."""
        if not self.producer or self.fallback is not None:
            return []
        return [
            Annotation(
                producer=self.producer,
                type=CONTEXTUALIZED,
                value={"original": self.original, "turns_used": self.turns_used},
                deterministic=False,
            )
        ]
