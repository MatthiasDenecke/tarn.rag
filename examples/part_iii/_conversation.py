"""The shared pieces of the Part III examples — one conversation, one narrator, one scripted model.

Both examples play the **same two-turn conversation** over the **same store**; only their config
differs (``conversation_memory.enabled``). So everything that is not the lesson lives here:

- :data:`TURNS` — the conversation and, per turn, the phrase and document that hold the answer.
- :class:`Narrator` — plays a turn and prints what happened: the query actually searched, a
  **model-free** check of where the passage holding the answer ranked among the passages the reader
  was shown, and the answer.
- :func:`scripted_model` — the offline stand-in for the LLM (see below).

**About the scripted model.** These examples run without an API key: ``run.py`` injects a
``StaticLanguageModel`` instead of a live LLM. So that the demonstration is not a foregone conclusion,
the script is bound to what the pipeline actually hands it and knows nothing else:

- as the **rewriter**, it resolves ``it`` to the most recent answer *in the transcript it is shown* —
  if memory did not deliver the earlier turn, it has nothing to substitute and returns the question
  unchanged;
- as the **reader**, it answers only from *the passages it is shown*, and only when one of them
  mentions what the question names — otherwise it says ``not specified``.

It proves the plumbing and the retrieval difference, not how a live model phrases a rewrite. The
verdict each example prints — the rank of the passage holding the answer — involves no model at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rich.console import Console as RichConsole
from rich.rule import Rule

from examples.common import corpus
from examples.part_ii._runner import Runner
from tarnrag import TarnRag
from tarnrag.console import Console as View
from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings
from tarnrag.core.resources.llm import Prompt, StaticLanguageModel
from tarnrag.generation import GenerationResult


@dataclass(frozen=True)
class Turn:
    """One scripted line of the conversation + where its answer lives in the corpus."""

    question: str
    gold: str  # the phrase a correct answer contains
    document: str  # the document whose passage holds it


# The conversation. Turn 2 is a FOLLOW-UP: "it" is the frame turn 1 just named. Its answer is in
# `lubrication-spec` — one of five near-identical frame lubrication specs in this store.
TURNS = (
    Turn("Which frame is the TX-200 built on?", gold="GMV", document="compressor-models"),
    Turn("Which oil grade does it require?", gold="ISO VG 68", document="lubrication-spec"),
)

# This series' store: corpus-2 plus the four distractor frame specs (the corpus Part II's rung 08 uses).
CORPORA = ("corpus-2", "corpus-2-frames")


@dataclass(frozen=True)
class Played:
    """What one turn did — returned by :meth:`Narrator.play` so a caller (the smoke test) can assert it."""

    asked: str  # the question as typed
    searched: str  # the query retrieval actually ran (== asked unless memory rewrote it)
    rank: int | None  # 1-based rank of the answer's passage among those the reader was shown; None = not shown
    result: GenerationResult


class Narrator:
    """Plays conversation turns through a :class:`~tarnrag.TarnRag` and narrates them. Reuses Part II's
    :class:`~examples.part_ii._runner.Runner` for the banner / corpus set-up and the console's own
    renderers for the answer and the history — one renderer, every surface."""

    def __init__(self, tarn: TarnRag, settings: Settings) -> None:
        self._tarn = tarn
        self._runner = Runner(tarn)
        self._out = RichConsole()
        # The depth the reader reads at (for the printout): the verdict is taken over the passages the
        # reader was actually shown, so "not in the top k" means exactly "the reader was never shown it".
        self._top_k = settings.components[GENERATION_PIPELINE]["reasoner"]["top_k"]

    def banner(self, title: str, **lines: str | None) -> None:
        self._runner.banner(title, **lines)

    async def ensure_corpus(self) -> None:
        await self._runner.ensure_corpus(*(corpus(name) for name in CORPORA))

    async def play(self, turn: Turn) -> Played:
        """One turn, as a single ``ask`` (so a follow-up is rewritten once, and everything printed is
        about that one run): what it searched — the standalone question the turn recorded, read back
        through ``history`` (as typed when memory is off) — then the model-free check of where the
        answer's passage ranked among the passages the reader was shown (the result's ``evidence``,
        in retrieval order), then the answer."""
        self._out.print(Rule(f"[bold cyan]you ›[/] {turn.question}", align="left", style="cyan"))

        result = (await self._tarn.ask(turn.question)).value
        recorded = (await self._tarn.history(last=1)).value  # empty when memory is off
        searched = recorded[-1].standalone_question if recorded else turn.question
        note = "as typed" if searched == turn.question else "rewritten against the conversation"
        self._out.print(f"[bold]searched[/]  {searched}  [dim]({note})[/]")
        documents = [passage.document_id for passage in result.evidence]
        self._out.print(f"[bold]top {self._top_k}[/]     " + "  ·  ".join(documents))
        rank = next(
            (i for i, passage in enumerate(result.evidence, 1) if turn.gold.lower() in passage.text.lower()),
            None,
        )
        if rank is None:
            self._out.print(
                f"[red]MISS[/]  the passage holding {turn.gold!r} ({turn.document}) is not in the "
                f"top {self._top_k} — the reader is never shown it"
            )
        else:
            self._out.print(f"[green]HIT [/]  the passage holding {turn.gold!r} ({turn.document}) is rank {rank}")
        self._out.print(View.create_answer_view(result))
        return Played(asked=turn.question, searched=searched, rank=rank, result=result)

    async def show_history(self) -> int:
        """Print the conversation as the store remembers it (or the facade's note that memory is off);
        return how many turns are remembered."""
        outcome = await self._tarn.history()
        self._out.print(Rule("[bold]what the conversation layer remembers[/]", align="left", style="cyan"))
        if outcome.value:
            self._out.print(View.create_history_view(outcome.value))
        for issue in outcome.report.issues:
            self._out.print(f"[yellow]{issue.severity.value}:[/] {issue.message}")
        return len(outcome.value)


# ---------------- the scripted model (see the module docstring) ----------------

_PRONOUN = re.compile(r"\bit\b")
_PASSAGE = re.compile(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)", re.MULTILINE | re.DOTALL)
# What a question asks for → how that fact is worded in a passage. First match wins, so the more
# specific "oil grade" is tried before "frame" (the rewritten follow-up mentions both).
_FACTS = (
    ("oil grade", re.compile(r"ISO VG \d+")),
    ("frame", re.compile(r"built on an? ((?:[A-Z][\w-]+ )+frame)")),
)
_NOT_SPECIFIED = json.dumps({"answer": "not specified", "steps": []})


def scripted_model() -> StaticLanguageModel:
    """The offline stand-in for the LLM: a rewriter and a reader, told apart by their prompts."""
    return StaticLanguageModel(_reply, name="part-iii-script")


def _reply(prompt: Prompt) -> str:
    if "standalone_question" in (prompt.system or ""):
        return _rewrite(prompt.user)
    return _read(prompt.user)


def _rewrite(user: str) -> str:
    """Resolve ``it`` in the latest question to the most recent answer in the transcript shown."""
    transcript, _, question = user.rpartition("Latest question: ")
    question = question.strip()
    # An answer line is "A: …", or "A (<marker>): …" when it was unverified / unchecked / declined.
    answers = re.findall(r"^\s+A(?: \([^)]*\))?: (.+)$", transcript, re.MULTILINE)
    if answers and _PRONOUN.search(question):
        question = _PRONOUN.sub(f"the {answers[-1].strip()}", question, count=1)
    return json.dumps({"standalone_question": question})


def _read(user: str) -> str:
    """Answer from the numbered passages shown — only from a passage that mentions what the question
    names (its capitalised / numbered terms). No named subject, or no such passage ⇒ not specified."""
    head, _, body = user.partition("\n\nPassages:\n")
    question = head.removeprefix("Question: ").strip()
    subject = [word.strip("?.,") for word in question.split()[1:] if re.search(r"[A-Z0-9]", word)]
    pattern = next((regex for cue, regex in _FACTS if cue in question.lower()), None)
    if not subject or pattern is None:
        return _NOT_SPECIFIED
    for number, text in _PASSAGE.findall(body):
        match = pattern.search(text)
        if match and all(term in text for term in subject):
            answer = match.group(match.lastindex or 0)
            sentence = _sentence_with(answer, text)
            return json.dumps({"answer": answer, "steps": [{"claim": sentence, "cited": [int(number)]}]})
    return _NOT_SPECIFIED


def _sentence_with(answer: str, passage: str) -> str:
    """The sentence of ``passage`` that states ``answer`` — the claim the scripted reader cites. Taken
    from the paragraph that holds it (so a heading line above is not glued onto the sentence)."""
    paragraph = next(p for p in re.split(r"\n\s*\n", passage) if answer in p)
    return next(s for s in re.split(r"(?<=\.)\s+", " ".join(paragraph.split())) if answer in s)
