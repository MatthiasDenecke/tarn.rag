"""TarnRag — the high-level facade over tarn.rag's ingestion + retrieval + generation engines.

A small, **output-free** session API over one store: ``ingest`` / ``docs`` / ``delete`` / ``retrieve`` /
``ask``. Each returns an :class:`~tarnrag.report.Outcome` — the value it produced, plus a
:class:`~tarnrag.report.Report` of any non-fatal issues encountered (empty when all went well). Nothing is
printed and nothing is silently skipped: a UI over the facade (the console, the tiqtasq REST API) renders
the value and surfaces the report. ``TarnRag`` is the composition root: it builds the one store (the
repository), the embedder, and the language model once and injects them into the engines that run over
them (one connection — retrieval sees ingests live, and works on an empty store; one embedding model; one
LLM client, shared by generation and any LLM-assisted retriever); the generation engine is built lazily on
the first ``ask``. It closes what it built, and only that. Use it directly::

    async with TarnRag("config.json") as tarn:
        ingested = await tarn.ingest(["path/to/your/files"])
        for issue in ingested.report.issues:
            ...  # e.g. a path that wasn't found
        answer = await tarn.ask("…")
        print(answer.value.answer)

…or behind a UI. ``llm`` may be injected (tests / a custom backend).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from tarnrag.contracts import CorpusStatus, RetrievalResult
from tarnrag.core.engine.config import Settings
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
from tarnrag.retrieval.types import SearchTrace
from tarnrag.storage.repository import DocumentRepository


class TarnRag:
    """An ingestion + retrieval + generation session over one store (see the module docstring)."""

    def __init__(self, settings: Settings | str | Path, llm: LanguageModel | None = None) -> None:
        """``settings`` is a ready ``Settings`` or the path to a JSON/YAML config to load (the ambient
        ``.env`` is ignored — the file is authoritative; OS env vars still supplement, e.g. the LLM key).
        ``llm`` may be injected (tests / a custom backend)."""
        if not isinstance(settings, Settings):
            settings = Settings.from_file(settings)
        self.settings = settings
        self._injected_llm = llm
        self._repository: DocumentRepository | None = None
        self._embedder: Embedder | None = None
        self._llm: LanguageModel | None = None  # the injected model, or the one open() builds
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
            # One LLM for the session — generation's reader AND any LLM-assisted retriever (hyde /
            # multi_query / llm_judge). Building it is free (the client connects on first use).
            self._llm = self._injected_llm or LanguageModel.create(self.settings.llm)
            # Establish/confirm the index identity (cheap, repository-only) so retrieval opens even on a
            # fresh store — without building the ingestion machinery, which waits until something ingests.
            await IngestionEngine.ensure_index_meta(self._repository, self._embedder)
            self._retrieval = await RetrievalEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder, llm=self._llm
            )
        except BaseException:
            await self.close()  # release the store we opened before re-raising (e.g. an index mismatch)
            raise
        return self

    async def close(self) -> None:
        """Release everything this facade built: the engines' own resources (the ingestion queue — the
        pgQueuer pool in distributed mode), the LLM (unless it was injected — an injected one is closed by
        its owner), the embedder's network client, and the one shared store. Idempotent."""
        if self._ingestion is not None:
            await self._ingestion.aclose()  # its queue; the store / embedder are ours, not its
        if self._retrieval is not None:
            await self._retrieval.aclose()  # it was handed everything — releases nothing of ours
        if self._llm is not None and self._injected_llm is None:
            await self._llm.aclose()
        if self._embedder is not None:
            self._embedder.close()
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
        ``search_text`` default."""
        return Outcome(await self._retrieval.search_text(query, top_k=top_k))

    async def explain(self, query: str, *, top_k: int = 8) -> Outcome[SearchTrace]:
        """Retrieval with its inner workings exposed — the same ranked results ``retrieve`` returns
        (``trace.results``) plus the :class:`~tarnrag.retrieval.types.SearchTrace` a UI renders to explain
        *why*: each retriever's candidates before fusion, the ranking at every pipeline stage
        (fused → merged → reranked → final) with per-component scores, and any routing decision.
        Output-free like the rest of the facade — it returns the data; a UI (the console's ``explain``)
        renders it."""
        return Outcome(await self._retrieval.explain_text(query, top_k=top_k))

    async def ask(self, query: str) -> Outcome[GenerationResult]:
        """Retrieval + generation — a grounded answer with a proof tree. Needs an LLM."""
        return Outcome(await self._gen().answer_text(query))

    def retrieval_context(self) -> RetrievalContext:
        """Everything the retrieval pipelines run against — the shared store + embedder, and the
        cross-encoder, license policy and LLM the session's retrieval engine holds — the seam the eval
        harness's ``sweep`` uses to score *alternative* pipeline specs over this one index (a spec with a
        reranker or an LLM-assisted retriever gets the same resources ``retrieve`` uses)."""
        return self._retrieval.context()

    async def _ingestion_engine(self) -> IngestionEngine:
        """The ingestion engine, built lazily on first use (ingest / docs / delete) — a read-only or eval
        session never spins up the orchestrator + queue + worker + sinks. Shares the open store."""
        if self._ingestion is None:
            self._ingestion = await IngestionEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder
            )
        return self._ingestion

    def _gen(self) -> GenerationEngine:
        """The generation engine, built once over the shared retrieval engine + the session's LLM (the
        pipeline comes from ``Settings`` via ``GenerationEngine.assemble``). Without an injected model, a
        missing API key fails here, fast and clearly — before any retrieval."""
        if self._generation is None:
            if self._injected_llm is None and not self.settings.llm.resolved_api_key():
                raise RuntimeError(
                    f"generation needs an LLM key — set {self.settings.llm.key_env_var()} "
                    "(the provider / model are in the config)"
                )
            self._generation = GenerationEngine.assemble(self._retrieval, self._llm, self.settings)
        return self._generation

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
