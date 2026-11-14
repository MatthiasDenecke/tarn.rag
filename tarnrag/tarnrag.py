"""TarnRag — the high-level facade over tarn.rag's ingestion + retrieval + generation engines.

A small, **output-free** session API over one store: ``ingest`` / ``docs`` / ``delete`` / ``retrieve`` /
``ask``. Each returns an :class:`~tarnrag.report.Outcome` — the value it produced, plus a
:class:`~tarnrag.report.Report` of any non-fatal issues encountered (empty when all went well). Nothing is
printed and nothing is silently skipped: a UI over the facade (the console, the tiqtasq REST API) renders
the value and surfaces the report. ``TarnRag`` is the composition root: it builds the one store (the
repository) and the embedder once and injects both into the engines that run over them (one connection —
retrieval sees ingests live, and works on an empty store); the generation engine is built lazily on the
first ``ask``. Use it directly::

    async with TarnRag("config.json") as tarn:
        ingested = await tarn.ingest(["path/to/your/files"])
        for issue in ingested.report.issues:
            ...  # e.g. a path that wasn't found
        answer = await tarn.ask("…")
        print(answer.value.answer)

…or behind a UI. ``llm`` may be injected (tests / a custom backend).

**Conversation memory** (opt-in — ``components.conversation_memory.enabled``; see
``doc/conversation-memory-design.md``). With it on, one ``TarnRag`` is one conversation: ``retrieve`` /
``explain`` / ``ask`` first make the question stand on its own against the last few turns (a follow-up
like "which oil grade does *it* require?" is rewritten by one LLM call), then run exactly as before on
that standalone question. ``ask`` and ``retrieve`` record a turn; ``explain`` is a dry run. The engines
never learn a conversation exists — this facade, the composition root, applies the layer in front of
them. ``history`` shows each question as typed and as understood; ``new_conversation`` / ``forget``
start over. With it off (the default) every call is stateless, exactly as without the layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Self

from bausatz import ComponentFactory

from tarnrag.contracts import ConversationTurn, CorpusStatus, RetrievalResult
from tarnrag.conversation import (
    Contextualization,
    ConversationContext,
    ConversationMemory,
)
from tarnrag.conversation.components.contextualizer import NO_LLM
from tarnrag.core.engine.config import CONVERSATION_MEMORY, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.generation.types import GenerationResult
from tarnrag.ingestion.components.extraction.load_parse import infer_source_kind
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.ingestion.engine.types import DocumentStatus, DocumentSummary
from tarnrag.report import Issue, Outcome, Report, Severity
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.types import Query, SearchTrace
from tarnrag.storage.repository import DocumentRepository


class TarnRag:
    """An ingestion + retrieval + generation session over one store (see the module docstring)."""

    def __init__(
        self,
        settings: Settings | str | Path,
        llm: LanguageModel | None = None,
        session_id: str | None = None,
    ) -> None:
        """``settings`` is a ready ``Settings`` or the path to a JSON/YAML config to load (the ambient
        ``.env`` is ignored — the file is authoritative; OS env vars still supplement, e.g. the LLM key).
        ``llm`` may be injected (tests / a custom backend). ``session_id`` names the conversation to
        join when conversation memory is enabled — resuming it if the store already holds its turns; it
        overrides the config's ``conversation_memory.session_id``, and with neither set each ``open``
        starts a fresh conversation. Ignored when memory is off (``session_id`` is then ``None``)."""
        if not isinstance(settings, Settings):
            settings = Settings.from_file(settings)
        self.settings = settings
        self._injected_llm = llm
        self._llm: LanguageModel | None = None  # the LLM this facade built (shared by ask + the rewrite)
        self._requested_session_id = session_id
        self._memory: ConversationMemory | None = None
        self._session_id: str | None = None  # the current conversation; None ⇔ memory is off
        # One conversation is sequential (design §4): with memory on, each read-run-record operation, and
        # a session switch, holds this lock, so a turn is always resolved against, and recorded in, the
        # conversation it started in, after the turns before it. Stateless calls never take it.
        self._turn_lock = asyncio.Lock()
        self._repository: DocumentRepository | None = None
        self._embedder: Embedder | None = None
        self._ingestion: IngestionEngine | None = None
        self._retrieval: RetrievalEngine | None = None
        self._generation: GenerationEngine | None = None

    async def open(self) -> TarnRag:
        """Build the shared resources once — the repository (the one store) and the passage/query
        embedder — plus the retrieval engine over them, and establish/confirm the index identity. The
        composition root: engines receive their dependencies. The ingestion engine is built lazily (on
        the first ingest / docs / delete) and generation on the first ask, so a read-only / eval session
        never spins up the ingestion machinery."""
        self._repository = await DocumentRepository.create(
            self.settings.database, self.settings.EMBEDDING_DIMENSION
        )
        try:
            self._embedder = Embedder.create(self.settings.embedding, self.settings.EMBEDDING_DIMENSION)
            # Establish/confirm the index identity (cheap, repository-only) so retrieval opens even on a
            # fresh store — without building the ingestion machinery, which waits until something ingests.
            await IngestionEngine.ensure_index_meta(self._repository, self._embedder)
            self._retrieval = await RetrievalEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder
            )
            # The conversation layer: built from its spec (validated here, so a bad one fails at open,
            # not on the first follow-up). Off by default — then there is no session and every call
            # below takes the stateless path untouched.
            self._memory = ComponentFactory.get().create_as(
                self.settings.components[CONVERSATION_MEMORY], ConversationMemory
            )
            if self._memory.enabled:
                self._session_id = (
                    self._requested_session_id
                    or self._memory.config.session_id
                    or ConversationMemory.new_session_id()
                )
        except BaseException:
            await self.close()  # release the store we opened before re-raising (e.g. an index mismatch)
            raise
        return self

    async def close(self) -> None:
        """Release everything this facade built: the generation LLM (unless it was injected — an
        injected one is closed by its owner), the ingestion engine (its queue — the pgQueuer pool in
        distributed mode), and the one shared store."""
        if self._injected_llm is None and self._llm is not None:
            await self._llm.aclose()  # the one LLM it built, shared by generation and the rewrite
        if self._ingestion is not None:
            await self._ingestion.aclose()  # the queue it built (+ the shared store, idempotently)
        if self._repository is not None:
            await self._repository.disconnect()  # the one store, shared by every engine

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ---------------- high-level API (each returns an Outcome[value] + Report) ----------------

    async def ingest(self, paths: list[str]) -> Outcome[list[DocumentStatus]]:
        """Ingest (or re-ingest) the given files / directories — a directory contributes the files in it.
        The document id is each file's stem, so re-ingesting a file upserts in place (under
        ``ID_POLICY='caller'``). The value is the resulting per-document statuses; the report flags paths
        that matched nothing, files skipped because their stem (= id) collides with another file's,
        replacements whose source kind differs from the stored document's (probably a different file
        sharing the stem), and documents that failed to ingest — nothing is dropped silently."""
        files, missing = self._expand(paths)
        issues = [Issue("not found", Severity.WARNING, subject=raw) for raw in missing]
        if self.settings.ID_POLICY == "caller":
            # The stem is the document id — two files sharing one stem (a.md + a.txt, or one name in two
            # directories) would silently upsert over each other. Keep the first, report the rest.
            files, collisions = self._split_stem_collisions(files)
            issues += [
                Issue(
                    f"skipped — its document id {skipped.stem!r} is already taken by {kept}",
                    Severity.ERROR,
                    subject=str(skipped),
                )
                for skipped, kept in collisions
            ]
            # Cross-call guard: an id that already exists is a legitimate re-ingest (upsert) — but if the
            # STORED source kind differs from the incoming file's, this is probably a different file that
            # happens to share the stem (a.pdf then a.md). The replace proceeds; the report says so.
            issues += await self._kind_change_warnings(files)
        if not files:
            return Outcome([], Report(tuple(issues)))
        ingestion = await self._ingestion_engine()
        source_ids = [p.stem for p in files] if self.settings.ID_POLICY == "caller" else None
        doc_ids = await ingestion.ingest_paths([str(p) for p in files], source_ids=source_ids)
        statuses = [s for s in [await ingestion.status(d) for d in doc_ids] if s is not None]
        issues += [
            Issue("ingestion failed", Severity.ERROR, subject=s.document_id)
            for s in statuses
            if s.status == "failed"
        ]
        return Outcome(statuses, Report(tuple(issues)))

    async def docs(self) -> Outcome[list[DocumentSummary]]:
        """Every ingested document (id + chunk / embedding counts)."""
        return Outcome(await (await self._ingestion_engine()).list_documents())

    async def status(self) -> Outcome[CorpusStatus]:
        """A snapshot of the document repository — document / chunk / embedding counts and the
        document-length distribution. Read-only: it queries the shared store directly, so it never builds
        the ingestion machinery."""
        return Outcome(await self._repository.corpus_stats())

    async def delete(self, document_id: str) -> Outcome[bool]:
        """Delete a document and everything derived from it. The value is False if it wasn't known."""
        return Outcome(await (await self._ingestion_engine()).delete_document(document_id))

    async def retrieve(self, query: str, *, top_k: int = 8) -> Outcome[list[RetrievalResult]]:
        """Retrieval only — the ranked, provenance-bearing passages. ``top_k`` mirrors the engine's
        ``search_text`` default. With conversation memory on, a follow-up is first made standalone
        against the conversation, and the question is recorded as a turn (no answer)."""
        async with self._conversation_turn() as session:
            question, issues = await self._contextualize(query, session)
            results = await self._retrieval.search(self._query(question, top_k=top_k))
            issues += await self._remember(session, question, kind="retrieve")
        return Outcome(results, Report(tuple(issues)))

    async def explain(self, query: str, *, top_k: int = 8) -> Outcome[SearchTrace]:
        """Retrieval with its inner workings exposed — the same ranked results ``retrieve`` returns
        (``trace.results``) plus the :class:`~tarnrag.retrieval.types.SearchTrace` a UI renders to explain
        *why*: each retriever's candidates before fusion, the ranking at every pipeline stage
        (fused → merged → reranked → final) with per-component scores, and any routing decision.
        Output-free like the rest of the facade — it returns the data; a UI (the console's ``explain``)
        renders it. With conversation memory on it is a **dry run**: the follow-up is made standalone by
        the same rewrite ``retrieve`` runs (``trace.query`` is the executed query and carries the original
        wording in its ``contextualized`` annotation), but nothing is recorded — a diagnostic never
        changes the conversation. The rewrite is its own LLM call, pinned to temperature 0 by default, so
        it normally matches what the next ``retrieve`` / ``ask`` runs; a hosted model doesn't strictly
        guarantee identical output."""
        async with self._conversation_turn() as session:
            question, issues = await self._contextualize(query, session)
            trace = await self._retrieval.explain(self._query(question, top_k=top_k))
        return Outcome(trace, Report(tuple(issues)))

    async def ask(self, query: str) -> Outcome[GenerationResult]:
        """Retrieval + generation — a grounded answer with a proof tree. Needs an LLM. With conversation
        memory on, a follow-up is first made standalone against the conversation (the reasoners, the
        grounding check and the proof tree then work on that question — the conversation itself never
        reaches the reader), and the turn is recorded with its answer and verdicts. An answer no
        grounding check verified is recorded with ``grounded=None`` — unchecked, not grounded."""
        generation = self._gen()  # first: no LLM ⇒ fail fast, before any read or retrieval
        async with self._conversation_turn() as session:
            question, issues = await self._contextualize(query, session)
            result = await generation.answer(self._query(question))
            checked = generation.verifies_grounding or result.abstained  # a refusal is a verdict either way
            issues += await self._remember(
                session, question, kind="ask", answer=result.answer,
                grounded=result.grounded if checked else None, abstained=result.abstained,
            )
        return Outcome(result, Report(tuple(issues)))

    # ---------------- conversation memory (opt-in; see the module docstring) ----------------

    @property
    def session_id(self) -> str | None:
        """The conversation this facade is in — ``None`` when conversation memory is off."""
        return self._session_id

    async def history(self, last: int | None = None) -> Outcome[list[ConversationTurn]]:
        """The current conversation, oldest first — each turn's question as typed
        (``question``), as understood (``standalone_question``), and for an ``ask`` its answer and
        verdicts. The whole stored transcript, not only the short-term window the rewriter reads;
        ``last`` keeps the most recent turns. This is how a UI shows what a follow-up was understood as."""
        if last is not None and last < 0:
            raise ValueError(f"last must be >= 0 (or None for the whole conversation), got {last}")
        if self._session_id is None:
            return Outcome([], self._memory_off_report())
        return Outcome(await self._memory.history(self._session_id, self._conversation_context(), last))

    async def new_conversation(self) -> Outcome[str | None]:
        """Start a fresh conversation: later calls no longer see the earlier turns. The value is the new
        session id. The previous conversation is **not** deleted — its transcript stays in the store
        (resumable by its id); use :meth:`forget` to remove turns."""
        if self._session_id is None:
            return Outcome(None, self._memory_off_report())
        async with self._turn_lock:  # after any in-flight turn has been recorded in the old conversation
            self._session_id = ConversationMemory.new_session_id()
            return Outcome(self._session_id)

    async def forget(self) -> Outcome[int]:
        """Delete the current conversation's turns from the store; the value is how many were removed.
        The conversation continues under the same id, with nothing remembered."""
        if self._session_id is None:
            return Outcome(0, self._memory_off_report())
        async with self._conversation_turn() as session:  # an in-flight turn can't reappear after it
            return Outcome(await self._memory.forget(session, self._conversation_context()))

    @asynccontextmanager
    async def _conversation_turn(self) -> AsyncIterator[str | None]:
        """One conversation-aware operation: yields the session it belongs to, read once at its start,
        while holding the turn lock, so the turn is resolved against and recorded in that same
        conversation and calls on one facade take turns. Memory off ⇒ yields ``None`` and takes no lock
        (stateless calls stay concurrent)."""
        if self._session_id is None:
            yield None
            return
        async with self._turn_lock:
            yield self._session_id

    async def _contextualize(self, question: str, session: str | None) -> tuple[Contextualization, list[Issue]]:
        """The question as it should be executed, plus any issue to report. Memory off (``session`` is
        ``None``) ⇒ the question as typed and nothing else happens (no store read, no LLM). A follow-up
        that could not be resolved is used as typed and **reported** — never silently passed through."""
        if session is None:
            return Contextualization(original=question, standalone=question), []
        result = await self._memory.contextualize(question, session, self._conversation_context())
        if result.fallback is None:
            return result, []
        hint = f" (set {self.settings.llm.key_env_var()})" if result.fallback == NO_LLM else ""
        message = (
            f"conversation memory could not resolve this follow-up — {result.fallback}{hint}; "
            "the question was used as typed"
        )
        return result, [Issue(message, Severity.WARNING, subject=question)]

    @staticmethod
    def _query(question: Contextualization, *, top_k: int = 8) -> Query:
        """The ``Query`` an engine receives: the standalone text, carrying the ``contextualized``
        annotation only when the question was rewritten (otherwise this is exactly the query the stateless
        path builds). Built here because the conversation package never imports ``retrieval``."""
        return Query(text=question.standalone, top_k=top_k, annotations=question.annotations())

    async def _remember(
        self,
        session: str | None,
        question: Contextualization,
        *,
        kind: str,
        answer: str | None = None,
        grounded: bool | None = None,
        abstained: bool | None = None,
    ) -> list[Issue]:
        """Record a completed operation as a turn in ``session`` (only reached when the operation
        returned — one that raised records nothing). A no-op with memory off. The operation's result is
        already computed, so a failed write doesn't discard it: it is reported as a WARNING instead (the
        next follow-up then can't see this turn)."""
        if session is None:
            return []
        turn = ConversationTurn(
            session_id=session,
            kind=kind,
            question=question.original,
            standalone_question=question.standalone,
            answer=answer,
            grounded=grounded,
            abstained=abstained,
        )
        try:
            await self._memory.remember(turn, self._conversation_context())
        except Exception as exc:  # noqa: BLE001 - optional memory must not cost the caller a finished answer
            message = (
                f"conversation memory could not record this turn — {type(exc).__name__}: {exc}; "
                "the result is complete, but a follow-up will not see this turn"
            )
            return [Issue(message, Severity.WARNING, subject=question.original)]
        return []

    def _conversation_context(self) -> ConversationContext:
        """The resources the conversation layer runs against: the one store (as the
        ``ConversationStore`` port) + the shared LLM, passed as its lazy builder — only a rewrite asks for
        it, so reading or recording turns, and a first turn with nothing to resolve, never build it."""
        return ConversationContext(self._repository, llm_factory=self._llm_or_none)

    @staticmethod
    def _memory_off_report() -> Report:
        return Report(
            (
                Issue(
                    "conversation memory is disabled — set components.conversation_memory.enabled: true",
                    Severity.WARNING,
                ),
            )
        )

    def retrieval_context(self) -> RetrievalContext:
        """The shared ``(store, embedder)`` the retrieval pipelines run against — the seam the eval
        harness's ``sweep`` uses to score *alternative* pipeline specs over this one index."""
        return RetrievalContext(self._repository, self._embedder)

    async def _ingestion_engine(self) -> IngestionEngine:
        """The ingestion engine, built lazily on first use (ingest / docs / delete) — a read-only or eval
        session never spins up the orchestrator + queue + worker + sinks. Shares the open store."""
        if self._ingestion is None:
            self._ingestion = await IngestionEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder
            )
        return self._ingestion

    def _gen(self) -> GenerationEngine:
        """The generation engine, built once over the shared retrieval engine + the configured LLM
        (the pipeline comes from ``Settings`` via ``GenerationEngine.assemble``)."""
        if self._generation is None:
            llm = self._llm_or_none()
            if llm is None:
                raise RuntimeError(
                    f"generation needs an LLM key — set {self.settings.llm.key_env_var()} "
                    "(the provider / model are in the config)"
                )
            self._generation = GenerationEngine.assemble(self._retrieval, llm, self.settings)
        return self._generation

    def _llm_or_none(self) -> LanguageModel | None:
        """The one LLM this session uses — for generation and for the conversation rewrite alike: the
        injected one, else the configured one built lazily on first need (no network until its first
        ``complete``), else ``None`` when no key is configured. Built at most once."""
        if self._injected_llm is not None:
            return self._injected_llm
        if self._llm is None and self.settings.llm.resolved_api_key():
            self._llm = LanguageModel.create(self.settings.llm)
        return self._llm

    async def _kind_change_warnings(self, files: list[Path]) -> list[Issue]:
        """A WARNING per file that is about to replace a stored document of a **different source kind**
        (the ``a.pdf`` then ``a.md`` case — same stem, so same id, but almost certainly a different
        document). A same-kind replace is the normal re-ingest contract and stays silent; a stored kind of
        ``""``/``"document"`` is the pre-fix unknown default, so it can't be compared and stays silent too.
        The replace itself proceeds either way — this is the report, not a refusal."""
        issues: list[Issue] = []
        for path in files:
            existing = await self._repository.get_document(path.stem)
            if existing is None:
                continue  # a fresh id — nothing is replaced
            stored = (existing.metadata or {}).get("source_kind") or ""
            incoming = infer_source_kind(str(path))
            if stored not in ("", "document") and incoming and stored != incoming:
                issues.append(
                    Issue(
                        f"replacing document {path.stem!r}, previously ingested from a {stored!r} "
                        f"source, with this {incoming!r} file",
                        Severity.WARNING,
                        subject=str(path),
                    )
                )
        return issues

    @staticmethod
    def _split_stem_collisions(files: list[Path]) -> tuple[list[Path], list[tuple[Path, Path]]]:
        """Partition ``files`` by stem uniqueness: the kept files (first file per stem, in order) and the
        collisions as ``(skipped, kept-it-collides-with)`` pairs. Only used under ``ID_POLICY='caller'``,
        where the stem is the document id — a collision would be a silent overwrite, so it is skipped and
        reported instead. Deterministic: ``_expand`` yields a stable order, so re-runs keep the same file."""
        first_by_stem: dict[str, Path] = {}
        kept: list[Path] = []
        collisions: list[tuple[Path, Path]] = []
        for path in files:
            existing = first_by_stem.get(path.stem)
            if existing is None:
                first_by_stem[path.stem] = path
                kept.append(path)
            else:
                collisions.append((path, existing))
        return kept, collisions

    @staticmethod
    def _expand(paths: list[str]) -> tuple[list[Path], list[str]]:
        """Resolve each path to file(s): a directory contributes the files it directly contains. Returns
        the resolved files and the raw paths that matched nothing (reported, not skipped silently)."""
        files: list[Path] = []
        missing: list[str] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                files.extend(sorted(p for p in path.iterdir() if p.is_file()))
            elif path.is_file():
                files.append(path)
            else:
                missing.append(raw)
        return files, missing
