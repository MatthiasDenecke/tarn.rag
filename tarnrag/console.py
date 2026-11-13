"""An interactive console for tarn.rag — ingest, query (retrieval), and ask (retrieval + generation).

Start it with one JSON config:

    python -m tarnrag.console path/to/config.json

The config is a ``Settings`` document — ``database`` / ``embedding`` / ``llm`` plus the
ingestion / retrieval / generation pipeline specs under ``components`` (a sample is at
``examples/console.config.json``). The API key is read from the environment (``ANTHROPIC_API_KEY``), not
the config. Then type commands at the ``tarn>`` prompt::

    ingest <path> ...    ingest (or RE-ingest) files; a directory ingests the files in it. The document
                         id is the filename stem, so re-ingesting a file replaces it.
    docs                 list ingested documents (id, chunks, embeddings)
    status               summarize the corpus — counts + document-length stats
    delete <id>          delete a document and everything derived from it
    retrieve <query>     retrieval only — the ranked passages
    explain <query>      retrieval with its inner workings — each retriever's candidates before fusion,
                         the ranking at every pipeline stage (with component scores), and any routing
    ask <query>          retrieval + generation — the grounded answer + its proof tree
    help                 show the commands
    quit                 exit

This is purely the **UI**: a rich parse-and-print REPL that owns all output and delegates the work to a
:class:`tarnrag.TarnRag` through its high-level methods. Rendering needs the ``console`` extra
(``pip install '.[console]'``). The result/trace render helpers are static factory methods
(``create_results_table`` / ``create_trace_view``), so a non-REPL caller (the example runner) can build the
same views without a live session.
"""

from __future__ import annotations

import asyncio
import sys
from typing import ClassVar

try:  # importing `readline` gives input() up/down history + line editing (Linux/macOS stdlib)
    import readline  # noqa: F401
except ImportError:  # pragma: no cover — e.g. Windows without pyreadline3; the REPL still works
    pass

from rich.console import Console as RichConsole
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from tarnrag.contracts import CorpusStatus, RetrievalResult
from tarnrag.generation.types import GenerationResult
from tarnrag.report import Report, Severity
from tarnrag.retrieval.types import RetrieverCandidates, SearchTrace
from tarnrag.tarnrag import TarnRag


class Console:
    """A rich REPL over a :class:`~tarnrag.TarnRag` session — all output lives here. Construct with
    an (opened) ``TarnRag`` and call :meth:`run`; the console never touches the engines directly, only the
    facade's high-level methods (``ingest`` / ``docs`` / ``delete`` / ``retrieve`` / ``explain`` /
    ``ask``). Owns its rich output sink (``self._out``) and the result/trace rendering."""

    _COMMANDS: ClassVar[list[tuple[str, str]]] = [
        ("ingest <path> ...", "ingest (or re-ingest) files; a directory ingests the files in it"),
        ("docs", "list the ingested documents"),
        ("status", "summarize the corpus — counts + document-length stats"),
        ("delete <id>", "delete a document and everything derived from it"),
        ("retrieve <query>", "retrieval only — the ranked passages"),
        ("explain <query>", "retrieval with its inner workings — per-retriever candidates, per-stage ranking"),
        ("ask <query>", "retrieval + generation — the grounded answer + its proof tree"),
        ("help", "show this list"),
        ("quit", "exit"),
    ]

    # The prompt is read with the builtin input() (so readline's history/editing applies); the colour is
    # raw ANSI wrapped in \001..\002 so readline counts only the visible width when it redraws the line.
    _PROMPT = "\001\033[1;36m\002tarn> \001\033[0m\002"

    def __init__(self, tarn: TarnRag) -> None:
        self._tarn = tarn
        self._out = RichConsole()
        self._prompt = self._PROMPT if sys.stdout.isatty() else "tarn> "  # plain when piped (no raw ANSI)

    async def run(self) -> None:
        """Read commands at the ``tarn>`` prompt until EOF / quit."""
        self._out.print(
            f"[bold]tarn.rag console[/]  [dim]{escape(self._tarn.settings.database.document_url)}[/]"
        )
        self._out.print("Type [cyan]help[/], or [cyan]quit[/] to exit.  [dim](↑/↓ for history)[/]\n")
        handlers = {
            "help": self._do_help,
            "ingest": self._do_ingest,
            "docs": self._do_docs,
            "status": self._do_status,
            "delete": self._do_delete,
            "retrieve": self._do_retrieve,
            "explain": self._do_explain,
            "ask": self._do_ask,
        }
        while True:
            try:
                line = (await asyncio.to_thread(input, self._prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                self._out.print()
                break
            if not line:
                continue
            command, _, arg = line.partition(" ")
            command = command.lower()
            if command in ("quit", "exit", "q"):
                break
            handler = handlers.get(command)
            if handler is None:
                self._out.print(f"[red]unknown command[/] {command!r} — type [cyan]help[/]")
                continue
            try:
                await handler(arg.strip())
            except Exception as exc:  # noqa: BLE001 - a REPL keeps the session alive on any command error
                self._out.print(f"[red]error:[/] {escape(str(exc))}")
        self._out.print("[dim]bye[/]")

    async def _do_help(self, _arg: str) -> None:
        table = Table(title="commands", show_edge=False, title_justify="left", header_style="bold")
        table.add_column("command", style="cyan", no_wrap=True)
        table.add_column("description", style="dim")
        for command, description in self._COMMANDS:
            table.add_row(command, description)
        self._out.print(table)

    async def _do_ingest(self, arg: str) -> None:
        if not arg:
            self._out.print("usage: ingest <path> ...", style="dim")
            return
        outcome = await self._tarn.ingest(arg.split())
        if outcome.value:
            table = Table(show_edge=False, header_style="bold")
            table.add_column("document", style="cyan")
            table.add_column("status")
            table.add_column("chunks", justify="right")
            table.add_column("embeddings", justify="right")
            for s in outcome.value:
                color = {"complete": "green", "failed": "red"}.get(s.status, "yellow")
                table.add_row(
                    s.document_id, f"[{color}]{s.status}[/]", str(s.chunk_count), str(s.embedding_count)
                )
            self._out.print(table)
        else:
            self._out.print("[yellow]nothing ingested[/]")
        self._render_report(outcome.report)

    async def _do_docs(self, _arg: str) -> None:
        outcome = await self._tarn.docs()
        if not outcome.value:
            self._out.print("[yellow]no documents — ingest some first[/]")
        else:
            table = Table(show_edge=False, header_style="bold")
            table.add_column("document", style="cyan")
            table.add_column("chunks", justify="right")
            table.add_column("embeddings", justify="right")
            for d in outcome.value:
                table.add_row(d.document_id, str(d.chunk_count), str(d.embedding_count))
            self._out.print(table)
            self._out.print(f"[dim]{len(outcome.value)} document(s)[/]")
        self._render_report(outcome.report)

    async def _do_status(self, _arg: str) -> None:
        outcome = await self._tarn.status()
        self._out.print(self.create_status_view(outcome.value))
        self._render_report(outcome.report)

    async def _do_delete(self, arg: str) -> None:
        if not arg:
            self._out.print("usage: delete <document-id>", style="dim")
            return
        outcome = await self._tarn.delete(arg)
        if outcome.value:
            self._out.print(f"[green]deleted[/] {escape(arg)}")
        else:
            self._out.print(f"[yellow]no such document:[/] {escape(arg)}")
        self._render_report(outcome.report)

    async def _do_retrieve(self, arg: str) -> None:
        if not arg:
            self._out.print("usage: retrieve <query>", style="dim")
            return
        outcome = await self._tarn.retrieve(arg)
        if not outcome.value:
            self._out.print("[yellow]no results — ingest some documents first[/]")
        else:
            self._out.print(self.create_results_table(outcome.value))
        self._render_report(outcome.report)

    async def _do_explain(self, arg: str) -> None:
        if not arg:
            self._out.print("usage: explain <query>", style="dim")
            return
        outcome = await self._tarn.explain(arg)
        if not outcome.value.results:
            self._out.print("[yellow]no results — ingest some documents first[/]")
        else:
            self._out.print(self.create_trace_view(outcome.value))
        self._render_report(outcome.report)

    async def _do_ask(self, arg: str) -> None:
        if not arg:
            self._out.print("usage: ask <query>", style="dim")
            return
        outcome = await self._tarn.ask(arg)
        self._out.print(self.create_answer_view(outcome.value))
        self._render_report(outcome.report)

    # ---------------- rendering (static factory methods — reusable without a live session) ----------------

    @staticmethod
    def create_results_table(
        results: list[RetrievalResult], *, prev_order: list[str] | None = None, title: str | None = None
    ) -> Table:
        """Build a ranked-results table: ``#``, the fused ``score``, one column per **component score**
        present (dense / sparse / cross_encoder / …), the document, and a passage snippet. When
        ``prev_order`` (the chunk-id order of the previous stage) is given, a ``Δ`` column shows how each
        row *moved* — what merging or reranking did. Backs the ``retrieve`` table and ``explain``'s
        per-stage tables, and is reusable by the example runner."""
        components: list[str] = []
        for r in results:
            for key in r.component_scores:
                if key not in components:
                    components.append(key)
        table = Table(
            title=title, show_edge=False, header_style="bold", title_justify="left", title_style="green"
        )
        table.add_column("#", justify="right", style="dim")
        if prev_order is not None:
            table.add_column("Δ", justify="center")
        table.add_column("score", justify="right")
        for key in components:
            table.add_column(key, justify="right", style="dim")
        table.add_column("document", style="cyan")
        table.add_column("passage")
        for rank, r in enumerate(results, 1):
            row = [str(rank)]
            if prev_order is not None:
                row.append(Console._movement(r.chunk_id, rank - 1, prev_order))
            row.append(f"{r.score:.3f}")
            row += [f"{r.component_scores[k]:.3f}" if k in r.component_scores else "" for k in components]
            row += [escape(r.document_id), escape(Console._snippet(r.text, 60))]
            table.add_row(*row)
        return table

    @staticmethod
    def create_trace_view(trace: SearchTrace) -> Group:
        """Build the full explain breakdown of a :class:`~tarnrag.retrieval.types.SearchTrace`: the query
        (and routing decision), the over-fetch / license-scope pre-filter in effect, each retriever's
        candidates before fusion, the ranking at every pipeline stage (``fused`` → ``merged`` →
        ``reranked`` → ``final``) with component scores + inter-stage movement, and a details table of the
        identity/provenance carried on each final result. Pure rendering — returns a renderable."""
        by_id = {r.chunk_id: r for stage in trace.stages for r in stage.results}
        parts: list = [Text.assemble(("query: ", "bold"), trace.query.text)]
        if trace.route_path:  # one hop per router passed through (nested routers each add theirs)
            hops = " → ".join(f"query_type={qt or '∅'} → route {route!r}" for qt, route in trace.route_path)
            parts.append(Text(f"routed: {hops}", style="magenta"))
        parts.append(Console._params_summary(trace.query))
        if trace.per_retriever:
            parts.append(Text("\nretrievers (before fusion)", style="bold"))
            parts += [Console._candidates_table(rc, by_id) for rc in trace.per_retriever]
        parts.append(Text("\npipeline stages", style="bold"))
        parts.append(Console._score_legend(trace.results))
        prev_order: list[str] | None = None
        for stage in trace.stages:
            parts.append(Console.create_results_table(stage.results, prev_order=prev_order, title=stage.name))
            prev_order = [r.chunk_id for r in stage.results]
        if trace.results:
            parts.append(Text("\nresult details", style="bold"))
            parts.append(Console.create_details_table(trace.results))
        return Group(*parts)

    @staticmethod
    def create_details_table(results: list[RetrievalResult]) -> Table:
        """The identity + provenance carried on each result but omitted from the ranking tables: the chunk
        id, document, section (its header path), locator, and license class. Most are richer once
        structure-aware chunking / enrichment are on; here they show what the pipeline produced beyond
        rank and score."""
        table = Table(show_edge=False, header_style="bold", title_justify="left", title_style="green")
        table.add_column("#", justify="right", style="dim")
        table.add_column("chunk", style="dim", no_wrap=True)
        table.add_column("document", style="cyan")
        table.add_column("section")
        table.add_column("locator")
        table.add_column("license", style="dim")
        for rank, r in enumerate(results, 1):
            header_path = r.provenance.header_path if r.provenance else []
            section = " › ".join(header_path) if header_path else "—"
            table.add_row(
                str(rank), r.chunk_id[:8], escape(r.document_id), escape(section),
                escape(r.locator or "—"), escape(r.license_class),
            )
        return table

    @staticmethod
    def _params_summary(query) -> Text:
        """A dim line: the over-fetch targets and the license/scope pre-filter the retrievers apply (which
        chunks a query may even see) — intermediate context the result tables don't show."""
        scope = "ALL" if query.scope == "ALL" else f"{len(query.scope)} method(s)"
        grounding = " · grounding-required" if query.permitted_filter().require_grounding else ""
        return Text(
            f"over-fetch: dense top-{query.dense_k}, sparse top-{query.sparse_k}  ·  "
            f"pre-filter: purpose={query.purpose.value}, scope={scope}{grounding}",
            style="dim",
        )

    @staticmethod
    def _score_legend(results: list[RetrievalResult]) -> Text:
        """Explain the numbers once: the fused ``score`` ranks the list (higher first), while each
        component column is that retriever's *raw* score, which may run the other way (dense is a cosine
        distance — lower is nearer)."""
        notes = {
            "dense": "cosine distance, lower = nearer",
            "sparse": "BM25, higher = stronger",
            "cross_encoder": "reranker relevance, higher = better",
        }
        components: list[str] = []
        for r in results:
            for key in r.component_scores:
                if key not in components:
                    components.append(key)
        legend = "  ·  ".join(f"{c}: {notes.get(c, 'raw score')}" for c in components)
        return Text(f"score = fused rank (higher first){'  ·  ' + legend if legend else ''}", style="dim")

    @staticmethod
    def create_answer_view(result: GenerationResult) -> Panel | Group:
        """Build the grounded-answer view: the answer panel (green when fully grounded, yellow otherwise)
        and — unless the model abstained — the proof tree, each claim marked ✓/✗ with its citations.
        Backs the ``ask`` command and is reusable by the example runner."""
        if result.abstained:
            return Panel(escape(result.answer), title="abstained", border_style="yellow")
        border = "green" if result.grounded else "yellow"
        label = "grounded" if result.grounded else "not fully grounded"
        panel = Panel(escape(result.answer), title=f"answer  ([{border}]{label}[/])", border_style=border)
        tree = Tree("[bold]proof[/]")
        for step in result.proof:
            mark = "[green]✓[/]" if step.grounded else "[red]✗[/]"
            node = tree.add(f"{mark} {escape(step.claim)}")
            for c in step.citations:
                node.add(f"[dim]cite[/] {escape(Console._cite(c))}")
        return Group(panel, tree)

    @staticmethod
    def create_status_view(status: CorpusStatus) -> Table:
        """Summarize the corpus in the document repository: document / chunk / embedding counts and the
        document-length distribution (min · median · mean · max, in characters)."""
        table = Table(
            title="corpus", show_edge=False, header_style="bold", title_justify="left", title_style="green"
        )
        table.add_column("metric", style="cyan")
        table.add_column("value", justify="right")
        table.add_row("documents", str(status.document_count))
        table.add_row("chunks", str(status.chunk_count))
        table.add_row("embeddings", str(status.embedding_count))
        table.add_row("chunks / doc (mean)", f"{status.mean_chunks_per_doc:.1f}")
        table.add_row(
            "doc length (chars)",
            f"min {status.min_chars} · median {status.median_chars:.0f} · "
            f"mean {status.mean_chars:.0f} · max {status.max_chars}",
        )
        table.add_row("total chars", f"{status.total_chars:,}")
        return table

    @staticmethod
    def _movement(chunk_id: str, index: int, prev_order: list[str]) -> str:
        """How a result moved from the previous stage: ``▲n`` up, ``▼n`` down, ``·`` unchanged, ``＋``
        newly appeared (e.g. an auto-merged section parent that wasn't a retrieved leaf)."""
        if chunk_id not in prev_order:
            return "[green]＋[/]"
        delta = prev_order.index(chunk_id) - index
        if delta > 0:
            return f"[green]▲{delta}[/]"
        if delta < 0:
            return f"[red]▼{-delta}[/]"
        return "[dim]·[/]"

    @staticmethod
    def _candidates_table(rc: RetrieverCandidates, by_id: dict[str, RetrievalResult]) -> Table:
        """One retriever's pre-fusion candidates — rank, the raw engine score (distance for dense, BM25
        for sparse), and the document/passage (resolved from the hydrated stage results when available)."""
        table = Table(
            title=f"{rc.key} · {len(rc.candidates)} candidates", show_edge=False, header_style="dim",
            title_justify="left", title_style="cyan",
        )
        table.add_column("rank", justify="right", style="dim")
        table.add_column("raw", justify="right")
        table.add_column("document", style="cyan")
        table.add_column("passage")
        for c in rc.candidates:
            hit = by_id.get(c.chunk_id)
            document = escape(hit.document_id) if hit else escape(c.chunk_id)
            passage = escape(Console._snippet(hit.text, 50)) if hit else "[dim]—[/]"
            table.add_row(str(c.rank), f"{c.raw_score:.3f}", document, passage)
        return table

    @staticmethod
    def _snippet(text: str, width: int = 90) -> str:
        text = " ".join(text.split())
        return text if len(text) <= width else text[: width - 1] + "…"

    @staticmethod
    def _cite(citation) -> str:
        return " > ".join(citation.header_path) or citation.locator or citation.document_id

    def _render_report(self, report: Report) -> None:
        """Show the issues a facade call reported — nothing when the report is empty (all went well)."""
        for issue in report.issues:
            color = "red" if issue.severity is Severity.ERROR else "yellow"
            subject = f"{escape(issue.subject)}: " if issue.subject else ""
            self._out.print(f"[{color}]{issue.severity.value}:[/] {subject}{escape(issue.message)}")


async def _main(config_path: str) -> None:
    async with TarnRag(config_path) as tarn:
        await Console(tarn).run()


def _load_dotenv() -> None:
    """Load a ``.env`` (searched from the cwd up) into the environment so secrets like the LLM key are
    picked up in this interactive session — a no-op without ``python-dotenv``. The config file stays
    authoritative (pydantic ranks the loaded YAML above env vars); this only supplies what it omits."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m tarnrag.console <config.json>")
    _load_dotenv()
    asyncio.run(_main(argv[0]))


if __name__ == "__main__":
    main()
