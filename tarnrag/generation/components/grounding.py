"""The GroundingChecker seam — verify each reasoning step's claim against its cited evidence.

A pure ``Component``: given a ``ReasonedAnswer`` (steps citing evidence by index + the evidence), it returns
a per-step ``Verdict`` — ``GROUNDED`` / ``UNGROUNDED`` / ``UNCERTAIN`` — for whether each claim is supported
by *its own* cited passages. The three-valued verdict is what lets verification **cascade**: a cheap checker
marks what it's confident about and leaves the rest ``UNCERTAIN`` for a more expensive one to resolve.
Backends, all config-driven:

- ``HeuristicGroundingChecker`` — content-word overlap; confident at the extremes, ``UNCERTAIN`` in the
  middle band (LLM-free, no extra resources, C++-portable);
- ``LLMGroundingChecker`` — a single batched verdict from the ``LanguageModel`` (catches paraphrase /
  contradiction the overlap misses) — decisive where the model answered, ``UNCERTAIN`` where it didn't;
- ``CascadingGroundingChecker`` — a container that runs child checkers in order and escalates only the
  ``UNCERTAIN`` steps to the next, so the expensive checker runs only on what the cheap one couldn't decide
  (and is skipped entirely when nothing is uncertain).

The ``GenerationPipeline`` runs the configured checker (optional), collapses each verdict to a grounded
flag, stamps it onto the proof tree, and applies the abstention policy. None configured ⇒ no verification.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import RetrievalResult
from bausatz import Component, ComponentFactory
from tarnrag.core.resources.llm import Prompt
from tarnrag.generation.components._parsing import extract_json
from tarnrag.generation.components._passages import passage_text
from tarnrag.generation.components.reasoner import ReasonedAnswer, ReasonedStep
from tarnrag.generation.context import GenerationContext


class Verdict(str, Enum):
    """A grounding verdict for one step. ``UNCERTAIN`` is the cascade hinge — a checker emits it when it
    can't decide, and a later checker resolves it."""

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    UNCERTAIN = "uncertain"


class GroundingChecker(Component):
    """Port: a per-step ``Verdict`` for a ``ReasonedAnswer`` — is each claim supported by its cited
    evidence? Returns a ``list[Verdict]`` aligned to ``reasoned.steps``."""

    class Config(Component.Config):
        """Base grounding-checker config; concrete checkers pin ``class_name``."""

        # How a cited TABLE chunk is rendered for verification (P1 PR-2) — keep it equal to the
        # reasoner's ``table_view`` so the fact-checker judges the same representation the reader
        # read. Evals pin this explicitly.
        table_view: Literal["structured", "text"] = "structured"

    @abstractmethod
    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[Verdict]:
        """One ``Verdict`` per step (aligned to ``reasoned.steps``)."""

    @staticmethod
    def _cited_passages(step: ReasonedStep, reasoned: ReasonedAnswer) -> list[RetrievalResult]:
        """The evidence a step cites, by index, in order."""
        return [reasoned.evidence[i] for i in step.cited]


# A compact stop set for the overlap heuristic (content words only — small + self-contained).
_STOPWORDS = frozenset(
    "a an the of to in on for and or but with at by from as into about than that this these those is are "
    "was were be been being it its do does did can could should would will may might has have had not no "
    "you your we our they their he she his her i".split()
)
_TOKEN = re.compile(r"\w+")


class HeuristicGroundingChecker(GroundingChecker):
    """LLM-free: score each claim by the fraction of its content words that appear in its cited passages,
    and be **honest about confidence** — overlap ``>= high_overlap`` ⇒ ``GROUNDED``, ``<= low_overlap`` ⇒
    ``UNGROUNDED``, and the murky middle ⇒ ``UNCERTAIN`` (which a cascade escalates). Cheap, deterministic,
    C++-portable; it can't tell paraphrase from coincidence, so it leaves those for the next checker."""

    class Config(GroundingChecker.Config):
        class_name: Literal["heuristic_grounding"] = "heuristic_grounding"
        high_overlap: float = 0.7  # >= this fraction of the claim's content words covered ⇒ GROUNDED
        low_overlap: float = 0.1  # <= ⇒ UNGROUNDED; strictly between the two ⇒ UNCERTAIN
        stopwords: list[str] | None = None  # None ⇒ built-in set

    config: HeuristicGroundingChecker.Config

    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[Verdict]:
        stop = frozenset(self.config.stopwords) if self.config.stopwords is not None else _STOPWORDS
        verdicts: list[Verdict] = []
        for step in reasoned.steps:
            claim_words = self._content_words(step.claim, stop)
            if not claim_words:
                verdicts.append(Verdict.UNGROUNDED)  # a claim with no content words can't be grounded
                continue
            cited_text = " ".join(
                passage_text(p, self.config.table_view) for p in self._cited_passages(step, reasoned)
            )
            overlap = len(claim_words & self._content_words(cited_text, stop)) / len(claim_words)
            if overlap >= self.config.high_overlap:
                verdicts.append(Verdict.GROUNDED)
            elif overlap <= self.config.low_overlap:
                verdicts.append(Verdict.UNGROUNDED)
            else:
                verdicts.append(Verdict.UNCERTAIN)
        return verdicts

    @staticmethod
    def _content_words(text: str, stop: frozenset[str]) -> set[str]:
        return {w for w in _TOKEN.findall(text.lower()) if len(w) > 1 and w not in stop}


_TRUE_WORDS = frozenset({"true", "yes", "supported", "grounded"})
_FALSE_WORDS = frozenset({"false", "no", "unsupported", "ungrounded", "not supported"})

_SYSTEM = (
    "You are a strict fact-checker. For each numbered claim, decide whether it is FULLY supported by its "
    "cited passages ALONE (no outside knowledge). Reply with a SINGLE JSON object and nothing else:\n"
    '{"verdicts": [true, false, ...]} — one boolean per claim, in order. true = supported.'
)


class LLMGroundingChecker(GroundingChecker):
    """Verify each claim against its cited passages with the ``LanguageModel`` — one batched call
    (``GROUNDED`` / ``UNGROUNDED`` per step). A step that cites nothing is ``UNGROUNDED`` without asking
    (no evidence can't support a claim — the heuristic checker's rule too). Where the reply gives no
    usable verdict for a step — unparseable, missing, or not a recognizable true/false — the verdict is
    ``UNCERTAIN``: the checker couldn't decide, so a cascade can escalate and the pipeline's
    ``uncertain_is_grounded`` policy decides (lenient by default — a verification hiccup still doesn't
    refuse an otherwise-good answer, but a strict deployment no longer gets a fake pass)."""

    class Config(GroundingChecker.Config):
        class_name: Literal["llm_grounding"] = "llm_grounding"

    config: LLMGroundingChecker.Config

    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[Verdict]:
        if not reasoned.steps:
            return []
        completion = await ctx.llm.complete(Prompt(system=_SYSTEM, user=self._format(reasoned)))
        verdicts = self._parse(completion.text, len(reasoned.steps))
        return [Verdict.UNGROUNDED if not s.cited else v for s, v in zip(reasoned.steps, verdicts)]

    def _format(self, reasoned: ReasonedAnswer) -> str:
        """The user message: each numbered claim + the passages it cites (rendered per ``table_view``,
        so the judge sees what the reader saw)."""
        blocks = []
        for n, step in enumerate(reasoned.steps, 1):
            cited = GroundingChecker._cited_passages(step, reasoned)
            passages = (
                "\n".join(f"  - {passage_text(p, self.config.table_view)}" for p in cited)
                if cited
                else "  (no passages cited)"
            )
            blocks.append(f"Claim {n}: {step.claim}\nCited passages:\n{passages}")
        return "\n\n".join(blocks)

    @staticmethod
    def _parse(text: str, n: int) -> list[Verdict]:
        """Parse ``{\"verdicts\": [true, false, ...]}`` into one verdict per step (truncated / padded to
        ``n``). Each entry is read strictly — a JSON boolean, a ``"true"`` / ``"false"``-style string (a
        model that quotes its booleans), 1 / 0, or an object carrying one of those under ``grounded`` /
        ``supported`` / ``verdict``; ``bool("false")`` is ``True``, so nothing is coerced loosely. An
        entry it can't read, a missing one, or an unparseable reply ⇒ ``UNCERTAIN``."""
        data = extract_json(text)
        raw = data.get("verdicts") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return [Verdict.UNCERTAIN] * n  # unparseable ⇒ undecided, not a pass
        out = [LLMGroundingChecker._verdict(v) for v in raw[:n]]
        return out + [Verdict.UNCERTAIN] * (n - len(out))  # no verdict for a step ⇒ undecided

    @staticmethod
    def _verdict(value: Any) -> Verdict:
        """One verdict entry, read strictly (see :meth:`_parse`)."""
        if isinstance(value, dict):
            value = next((value[k] for k in ("grounded", "supported", "verdict") if k in value), None)
        if isinstance(value, bool):
            return Verdict.GROUNDED if value else Verdict.UNGROUNDED
        if isinstance(value, int | float) and value in (0, 1):
            return Verdict.GROUNDED if value == 1 else Verdict.UNGROUNDED
        if isinstance(value, str):
            word = value.strip().lower()
            if word in _TRUE_WORDS:
                return Verdict.GROUNDED
            if word in _FALSE_WORDS:
                return Verdict.UNGROUNDED
        return Verdict.UNCERTAIN


class CascadingGroundingChecker(GroundingChecker):
    """A container that runs child checkers in order and escalates only the steps still ``UNCERTAIN`` to the
    next — so the cheap checker resolves what it can and the expensive one (e.g. the LLM) adjudicates just
    the leftovers, and is skipped entirely when nothing is uncertain. Usual config: ``[heuristic, llm]``."""

    class Config(GroundingChecker.Config):
        class_name: Literal["cascading_grounding"] = "cascading_grounding"
        checkers: list[GroundingChecker.Config] = Field(default_factory=list)  # child checkers, cheap → expensive

    config: CascadingGroundingChecker.Config

    def __init__(self, config: CascadingGroundingChecker.Config) -> None:
        super().__init__(config)
        self._checkers: list[GroundingChecker] = []

    def _build_children(self, factory: ComponentFactory) -> None:
        self._checkers = [factory.build_as(cfg, GroundingChecker) for cfg in self.config.checkers]

    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[Verdict]:
        self.ensure_built()
        verdicts: list[Verdict] = [Verdict.UNCERTAIN] * len(reasoned.steps)
        for checker in self._checkers:
            pending = [i for i, v in enumerate(verdicts) if v is Verdict.UNCERTAIN]
            if not pending:
                break  # everything resolved — skip the remaining (more expensive) checkers
            subset = ReasonedAnswer(reasoned.answer, [reasoned.steps[i] for i in pending], reasoned.evidence)
            resolved = await checker.check(subset, ctx)
            for j, i in enumerate(pending):
                verdicts[i] = resolved[j]
        return verdicts
