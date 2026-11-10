"""The Reasoner seam — retrieve, read, and produce an answer with supporting reasoning steps.

A ``Reasoner`` is a config-driven ``Component`` (a pure strategy; resources injected via
``GenerationContext``). It returns a ``ReasonedAnswer`` (answer + steps citing evidence *by index* + the
evidence list); the ``EvidenceAssembler`` turns those cited indices into provenance-bearing ``Citation``s.
Three strategies share the seam (swap the reasoner spec):

- ``SingleHopReasoner`` — one retrieval round + one read (the MVP).
- ``IterativeReasoner`` — multi-hop by a retrieve↔read loop: the model answers or asks for a follow-up
  search; evidence accumulates until it's confident or ``max_hops`` is reached (handles dependent chains).
- ``DecompositionReasoner`` — multi-hop by splitting the question into independent sub-questions up front,
  retrieving for each, then synthesizing one answer.

The passage-formatting / cited-index parsing / read logic live on the base, so all three reuse them.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from bausatz import Component

from tarnrag.contracts import RetrievalResult
from tarnrag.core.resources.llm import Prompt
from tarnrag.generation.components._parsing import extract_json
from tarnrag.generation.components._passages import passage_text
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


@dataclass(frozen=True)
class ReasonedStep:
    """One reasoning step before provenance is attached: a claim + the evidence indices it cites."""

    claim: str
    cited: list[int] = field(default_factory=list)  # indices into ``ReasonedAnswer.evidence``


@dataclass
class ReasonedAnswer:
    """A ``Reasoner``'s output: the answer, the steps (citing evidence by index), and the evidence.
    ``abstained`` marks a refusal made *before* the read (the answerability gate) — the pipeline
    surfaces it as an abstention without running verification."""

    answer: str
    steps: list[ReasonedStep]
    evidence: list[RetrievalResult]
    abstained: bool = False


# The read prompt — answer using ONLY the numbered passages, citing per claim. Shared by the single-hop
# read and the decomposition synthesis (both are "read these passages → answer + cited steps"). The answer
# is demanded as the MINIMAL span (multi-hop QA is scored by token-F1/EM against short gold answers — a
# narrated sentence tanks both even when correct; see doc/phases.md Phase 0).
_READ_SYSTEM = (
    "You are a careful technical assistant. Answer the question using ONLY the numbered passages "
    "provided; do not use outside knowledge. If the passages do not contain the answer, say so plainly.\n\n"
    "Reply with a SINGLE JSON object and nothing else:\n"
    '{"answer": "<shortest answer span>", "steps": [{"claim": "<one supported statement>", '
    '"cited": [<passage numbers this claim relies on>]}]}\n'
    'Make "answer" the MINIMAL span that answers the question — a name, entity, number, date, or yes/no '
    '(e.g. "Eiffel Tower", "1889", "yes"); no sentence, no explanation, no restating the question. Put '
    "reasoning in the steps, not in the answer.\n"
    'Each claim\'s "cited" lists the 1-based passage numbers that support it; use only numbers shown.'
)

# A leading "the answer is …" / "answer:" the model sometimes prepends — stripped so it doesn't count as
# answer tokens (surrounding quotes too). Trailing punctuation/articles are left to the scorer's normalization.
_ANSWER_LEADIN = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer(?:\s+is\b|\s*:)\s*", re.IGNORECASE)


class Reasoner(Component):
    """Port: produce a ``ReasonedAnswer`` for a query over the ``GenerationContext`` (retrieve + read)."""

    class Config(Component.Config):
        """Base reasoner config; concrete reasoners pin ``class_name``."""

        # How a TABLE chunk is laid before the model (P1 PR-2): 'structured' renders it from its cells
        # (values bound to headers — see ``_passages``); 'text' is the raw stored grid (the pre-P1
        # behavior). Evals pin this explicitly — a measurement never inherits the default.
        table_view: Literal["structured", "text"] = "structured"

    @abstractmethod
    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        """Retrieve, read, and return the answer + supporting steps + the evidence drawn on."""

    # ---------------- shared helpers (used by every reasoner) ----------------

    def _numbered_passages(self, results: list[RetrievalResult]) -> str:
        """The retrieved passages, numbered 1-based for citation (``(none retrieved)`` when empty) —
        each rendered per the configured ``table_view`` (the shared ``passage_text``)."""
        if not results:
            return "(none retrieved)"
        return "\n".join(
            f"[{i + 1}] {Reasoner._label(r)}{passage_text(r, self.config.table_view)}"
            for i, r in enumerate(results)
        )

    @staticmethod
    def _label(result: RetrievalResult) -> str:
        """A light source hint for the model — the chunk's header path or locator, if any."""
        prov = result.provenance
        if prov and prov.header_path:
            return " > ".join(prov.header_path) + " — "
        return f"{result.locator} — " if result.locator else ""

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Trim a final answer toward its span: drop a leading 'the answer is …' / 'answer:' lead-in and any
        surrounding quotes the model adds (multi-hop QA wants the minimal span, not a sentence). Trailing
        punctuation / articles are left to the scorer's SQuAD normalization."""
        a = answer.strip().strip('"').strip("'").strip()
        return _ANSWER_LEADIN.sub("", a).strip() or a

    @staticmethod
    def _read(text: str, n: int) -> tuple[str, list[ReasonedStep]]:
        """Parse a read reply ``{answer, steps:[{claim, cited}]}``; fall back to ``(text, cite-all)``."""
        data = extract_json(text)
        if not isinstance(data, dict) or "answer" not in data:
            answer = Reasoner._clean_answer(text)
            return answer, [ReasonedStep(claim=answer, cited=list(range(n)))]
        answer = Reasoner._clean_answer(str(data.get("answer", "")))
        return answer, Reasoner._parse_steps(data.get("steps"), answer, n)

    @staticmethod
    def _parse_steps(raw_steps: Any, answer: str, n: int) -> list[ReasonedStep]:
        """Parse a ``steps`` array into ``ReasonedStep``s with validated cited indices. An explicit empty
        array is the model saying it cites nothing — e.g. "the passages don't say" — so the answer
        becomes one step citing **nothing** (verification then grades it honestly) rather than one
        fabricated to cite every passage. A missing or malformed ``steps`` (a format slip) still falls
        back to one cite-all step over ``answer``, so a well-formed answer always carries its evidence."""
        if isinstance(raw_steps, list) and not raw_steps:
            return [ReasonedStep(claim=answer, cited=[])]
        steps: list[ReasonedStep] = []
        for s in raw_steps or []:
            if not isinstance(s, dict):
                continue
            claim = str(s.get("claim", "")).strip()
            if claim:
                steps.append(ReasonedStep(claim=claim, cited=Reasoner._valid_indices(s.get("cited"), n)))
        return steps or [ReasonedStep(claim=answer, cited=list(range(n)))]

    @staticmethod
    def _valid_indices(cited: Any, n: int) -> list[int]:
        """Map 1-based passage numbers to in-range 0-based evidence indices — dropped if out of range,
        deduped, order preserved (bools rejected, since ``True``/``False`` are ints in Python)."""
        if not isinstance(cited, list):
            return []
        out: list[int] = []
        for c in cited:
            if isinstance(c, bool) or not isinstance(c, int):
                continue
            i = c - 1
            if 0 <= i < n and i not in out:
                out.append(i)
        return out

    @staticmethod
    def _accumulate(
        evidence: list[RetrievalResult], seen: set[str], results: list[RetrievalResult]
    ) -> None:
        """Append ``results`` to ``evidence`` in order, skipping chunk ids already gathered — the
        dedup multi-hop reasoners share as they pool passages across retrieval rounds. Mutates both
        ``evidence`` and ``seen``."""
        for r in results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                evidence.append(r)


class SingleHopReasoner(Reasoner):
    """One retrieval round + one read — the MVP reasoner. Search → numbered passages → strict-JSON answer
    with per-claim citations. Falls back to citing all retrieved passages if the reply isn't parseable
    JSON, so an answer always carries its evidence."""

    class Config(Reasoner.Config):
        class_name: Literal["single_hop"] = "single_hop"
        top_k: int = 8  # passages to retrieve and lay before the model

    config: SingleHopReasoner.Config

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        results = await ctx.retrieval.search(replace(query, top_k=self.config.top_k))
        user = f"Question: {query.text}\n\nPassages:\n{self._numbered_passages(results)}"
        completion = await ctx.llm.complete(Prompt(system=_READ_SYSTEM, user=user))
        answer, steps = self._read(completion.text, len(results))
        return ReasonedAnswer(answer=answer, steps=steps, evidence=results)


_ITER_SYSTEM = (
    "You research a question over a corpus you can search in steps. Given the question and the passages "
    "gathered SO FAR (numbered), decide whether they suffice — use ONLY these passages.\n\n"
    "Reply with a SINGLE JSON object and nothing else:\n"
    '{"done": <true|false>, "answer": "<shortest answer span, if done>", '
    '"follow_up": "<one focused search query, if not done>", '
    '"steps": [{"claim": "<supported statement>", "cited": [<passage numbers>]}]}\n'
    'When done, "answer" is the MINIMAL span — a name, entity, number, date, or yes/no (e.g. "Eiffel Tower", '
    '"1889", "yes"); no sentence, no explanation. '
    "When the passages are enough, set done=true with answer + steps. Otherwise done=false with a follow_up "
    "search that would fill the gap. On the final step you must answer with what you have."
)


class IterativeReasoner(Reasoner):
    """Multi-hop by an iterative retrieve↔read loop: each step searches, the model either answers or asks
    for a follow-up search, evidence accumulates, until it's confident or ``max_hops`` is reached. Handles
    dependent chains — a later hop can build on what an earlier one found."""

    class Config(Reasoner.Config):
        class_name: Literal["iterative"] = "iterative"
        max_hops: int = 3
        top_k: int = 8  # passages retrieved per hop

    config: IterativeReasoner.Config

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        evidence: list[RetrievalResult] = []
        seen: set[str] = set()
        search_query = query.text
        for hop in range(self.config.max_hops):
            results = await ctx.retrieval.search(replace(query, text=search_query, top_k=self.config.top_k))
            self._accumulate(evidence, seen, results)
            completion = await ctx.llm.complete(
                Prompt(
                    system=_ITER_SYSTEM,
                    user=f"Question: {query.text}\n\nPassages so far:\n{self._numbered_passages(evidence)}",
                )
            )
            data = extract_json(completion.text)
            if not isinstance(data, dict):  # unparseable ⇒ answer with the raw text, citing all gathered
                answer = self._clean_answer(completion.text)
                return ReasonedAnswer(answer, [ReasonedStep(answer, list(range(len(evidence))))], evidence)
            follow_up = str(data.get("follow_up", "")).strip()
            if hop == self.config.max_hops - 1 or data.get("done") or not follow_up:
                answer = self._clean_answer(str(data.get("answer", "")))
                return ReasonedAnswer(answer, self._parse_steps(data.get("steps"), answer, len(evidence)), evidence)
            search_query = follow_up
        return ReasonedAnswer("", [], evidence)  # only reached if max_hops <= 0


_DECOMPOSE_SYSTEM = (
    "Break the question into a few INDEPENDENT sub-questions whose answers together answer it. Keep them "
    'focused and few. Reply with a SINGLE JSON object and nothing else: {"sub_questions": ["...", "..."]}.'
)


class DecompositionReasoner(Reasoner):
    """Multi-hop by decomposition: the model splits the question into independent sub-questions up front,
    each is retrieved for, and the gathered evidence is synthesized into one answer. Good for questions
    that gather several independent facts (the sub-queries don't depend on each other)."""

    class Config(Reasoner.Config):
        class_name: Literal["decomposition"] = "decomposition"
        max_subquestions: int = 4
        top_k: int = 8  # passages retrieved per sub-question

    config: DecompositionReasoner.Config

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        evidence: list[RetrievalResult] = []
        seen: set[str] = set()
        for sub_question in await self._decompose(query.text, ctx):
            results = await ctx.retrieval.search(replace(query, text=sub_question, top_k=self.config.top_k))
            self._accumulate(evidence, seen, results)
        completion = await ctx.llm.complete(
            Prompt(
                system=_READ_SYSTEM,
                user=f"Question: {query.text}\n\nPassages:\n{self._numbered_passages(evidence)}",
            )
        )
        answer, steps = self._read(completion.text, len(evidence))
        return ReasonedAnswer(answer=answer, steps=steps, evidence=evidence)

    async def _decompose(self, question: str, ctx: GenerationContext) -> list[str]:
        """The sub-questions to retrieve for — falls back to the original question if none parse."""
        completion = await ctx.llm.complete(Prompt(system=_DECOMPOSE_SYSTEM, user=f"Question: {question}"))
        data = extract_json(completion.text)
        raw = data.get("sub_questions") if isinstance(data, dict) else None
        subs = [str(s).strip() for s in raw if str(s).strip()] if isinstance(raw, list) else []
        return subs[: self.config.max_subquestions] or [question]
