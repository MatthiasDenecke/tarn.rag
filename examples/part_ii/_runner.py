"""The thin teaching shell for the Part II examples — the narration around a YAML config.

Each example *is* a YAML config (run it directly with ``python -m tarnrag.console cfg.yaml``); the
:class:`Runner` is the scripted story around it. An example's ``run.py`` opens a
:class:`~tarnrag.TarnRag` over its config, wraps it in a ``Runner``, ingests the shared corpus once, then
runs a *good* and a *bad* probe through retrieval (showing the ``explain`` breakdown) and — in the
generation act — through ``ask``, marking each probe HIT/MISS against a gold phrase so the failure a step
demonstrates, and the next step's fix, is **shown**, not asserted.

The rendering is the console's own static factory methods (``Console.create_trace_view`` /
``create_results_table`` / ``create_answer_view``) — one renderer, two surfaces: the REPL and these
examples. ``Runner`` adds only the teaching orchestration (it stays in the examples layer, out of the
library). Needs the ``console`` extra (rich).
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.rule import Rule

from tarnrag import TarnRag
from tarnrag.console import Console as View
from tarnrag.generation import GenerationResult


class Runner:
    """The teaching shell for one Part II example: drives a :class:`~tarnrag.TarnRag` through good/bad
    probes and narrates the result. Construct with the open session and its own rich output sink; reuses
    the console's static renderers for the actual rendering."""

    def __init__(self, tarn: TarnRag) -> None:
        self._tarn = tarn
        self._out = RichConsole()

    def note(self, text: str) -> None:
        """Print a dim one-line observation (e.g. a count) alongside the rendered probes."""
        self._out.print(f"[dim]→ {text}[/]")

    def banner(self, title: str, *, shows: str, fails: str | None = None, fixed_next: str | None = None) -> None:
        """Frame a step: what it demonstrates, what still fails here, and what the next step changes to
        fix it. The three lines are the spine of the ladder — every step has a ``shows``; an intermediate
        step also has a ``fails`` and the ``fixed_next`` that repairs it."""
        lines = [f"[bold]shows[/]  {shows}"]
        if fails:
            lines.append(f"[yellow]fails[/]  {fails}")
        if fixed_next:
            lines.append(f"[green]next [/]  {fixed_next}")
        self._out.print(Panel("\n".join(lines), title=title, title_align="left", border_style="cyan"))

    async def ingest_corpus(self, *paths: str | Path) -> None:
        """Ingest the corpus into the session's store (re-ingesting is idempotent) and print a one-line
        summary, surfacing any issues the facade reported (a path that matched nothing, a failed
        document)."""
        outcome = await self._tarn.ingest([str(p) for p in paths])
        chunks = sum(d.chunk_count for d in outcome.value)
        self._out.print(f"[dim]ingested {len(outcome.value)} documents · {chunks} chunks[/]")
        for issue in outcome.report.issues:
            self._out.print(f"[yellow]{issue.severity.value}[/] {issue.subject}: {issue.message}")

    async def ensure_corpus(self, *paths: str | Path) -> None:
        """Ingest the corpus only if the store is empty — so the shared base store is built once (the
        console flow / Example 00), while a standalone example run still works. Idempotent either way."""
        docs = (await self._tarn.docs()).value
        if docs:
            self._out.print(f"[dim]corpus already present · {len(docs)} documents[/]")
        else:
            await self.ingest_corpus(*paths)

    async def show_retrieval(
        self, query: str, *, gold: str | None = None, label: str | None = None, top_k: int = 8
    ) -> bool | None:
        """Run ``explain`` for one probe and render its inner workings (per-retriever candidates, the
        ranking at each stage, component scores, routing). When a ``gold`` phrase is given, print a
        HIT/MISS verdict — a hit is content-based: the phrase appears in some top-``k`` passage — and
        return it (``None`` when no gold is given). This is the label-free check the whole ladder turns
        on."""
        self._out.print(Rule(self._probe_title(query, label), align="left", style="cyan"))
        trace = (await self._tarn.explain(query, top_k=top_k)).value
        self._out.print(View.create_trace_view(trace))
        if gold is None:
            return None
        ok = any(gold.lower() in result.text.lower() for result in trace.results)
        verdict = "[green]HIT [/]" if ok else "[red]MISS[/]"
        self._out.print(f"{verdict} gold phrase {gold!r} in the top {top_k}")
        return ok

    async def show_answer(self, query: str, *, gold: str | None = None, label: str | None = None) -> GenerationResult:
        """Run ``ask`` for one probe and render the grounded answer + its proof tree (needs the LLM key in
        ``OPENAI_LLM_KEY``). With a ``gold`` phrase, also print a HIT/MISS verdict (the phrase appears in the
        answer, case-insensitive). Returns the full result so a caller can inspect ``grounded`` /
        ``abstained``. Used by the generation act of the ladder."""
        self._out.print(Rule(self._probe_title(query, label), align="left", style="cyan"))
        result = (await self._tarn.ask(query)).value
        self._out.print(View.create_answer_view(result))
        if gold is not None:
            ok = gold.lower() in result.answer.lower()
            self._out.print(f"{'[green]HIT [/]' if ok else '[red]MISS[/]'} gold phrase {gold!r} in the answer")
        return result

    def show_result(self, result: GenerationResult, *, label: str) -> GenerationResult:
        """Render an already-computed generation result — the rendering half of :meth:`show_answer`, for a
        result that wasn't produced by a live ``ask`` (e.g. a constructed illustration). Returns it."""
        self._out.print(Rule(f"[bold]{label}[/]", align="left", style="cyan"))
        self._out.print(View.create_answer_view(result))
        return result

    @staticmethod
    def _probe_title(query: str, label: str | None) -> str:
        return f"[bold]{label}[/]  {query}" if label else query
