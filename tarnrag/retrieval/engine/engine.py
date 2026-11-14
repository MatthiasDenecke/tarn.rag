"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

A thin async facade over the §8 repository: it validates index compatibility (``open()`` — the one place
checked, refusing an index built with a different embedding pipeline (fingerprint) or schema), builds the
``Searcher`` from ``Settings`` (the ``RETRIEVAL_PIPELINE`` spec — a ``RetrievalPipeline`` (**hybrid** by
default from ``Settings``: dense + sparse, RRF-fused; a bare spec is the minimal dense-only building
block) or a ``RoutingRetrievalPipeline``), and delegates ``search`` to it. **Comparing retrieval
methods = varying the ``RETRIEVAL_PIPELINE`` spec.**
"""

from __future__ import annotations

from typing import Any

from bausatz import ComponentFactory

from tarnrag.contracts import IndexMeta, RetrievalResult
from tarnrag.core.engine.config import (
    LICENSE_POLICY,
    RETRIEVAL_PIPELINE,
    Settings,
    get_settings,
)
from tarnrag.core.engine.engine import Engine
from tarnrag.core.exceptions import RetrievalError
from tarnrag.core.resources.cross_encoder import CrossEncoder, OnnxCrossEncoder
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.retrieval.components.license_policy import LicensePolicy
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query, SearchTrace
from tarnrag.storage.repository import DocumentRepository

_DEFAULT_PIPELINE: dict[str, Any] = {"class_name": "retrieval_pipeline"}  # dense + identity fuser
_DEFAULT_LICENSE_POLICY: dict[str, Any] = {"class_name": "default_license"}  # ModusQ §5.6


class RetrievalEngine(Engine):
    """
    Async query facade over the §8 repository. Built from ``Settings`` via ``create`` / ``open`` (which
    validate index compatibility); the bare constructor is the data-holder seam — inject a pre-built
    ``Searcher`` (tests / advanced wiring). Delegates ``search`` to a config-driven ``RetrievalPipeline``
    (retrieve → fuse → hydrate → assemble); ``search`` / ``search_text`` are async — they await the store.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: Embedder,
        searcher: Searcher,
        cross_encoder: CrossEncoder | None = None,
        license_policy: LicensePolicy | None = None,
        llm: LanguageModel | None = None,
        *,
        owns_llm: bool = False,
    ):
        """Pure data-holder: inject the resources (the read store + query embedder + an optional reranker +
        an optional license policy + an optional ``llm`` for the bridge components) and the pre-built
        ``Searcher``. Build from ``Settings`` via :meth:`open` / :meth:`create`. Injected resources stay
        their owner's: ``aclose`` releases only what this engine built (``owns_llm`` / the store and
        embedder ``create`` built)."""
        self.repository = repository
        self.embedder = embedder
        self._pipeline = searcher
        self._cross_encoder = cross_encoder
        self._license_policy = license_policy
        self._llm = llm
        self._owns_llm = owns_llm
        self._owns_embedder = False  # set by create() when it built the embedder

    @staticmethod
    def build_searcher(settings: Settings | None = None) -> Searcher:
        """Build the retrieval ``Searcher`` from the ``RETRIEVAL_PIPELINE`` spec — the retrieval analog of
        ``IngestionEngine.build_pipeline``. A ``Settings``-driven build defaults to HYBRID (dense + sparse,
        RRF — filled by ``Settings``); the settings-less fallback stays the minimal dense-only pipeline
        (the low-level seam makes no composition choice)."""
        spec = settings.components.get(RETRIEVAL_PIPELINE) if settings is not None else None
        return ComponentFactory.get().create_as(spec or _DEFAULT_PIPELINE, Searcher)

    @staticmethod
    def build_license_policy(settings: Settings | None = None) -> LicensePolicy | None:
        """Build the ``LicensePolicy`` from the ``LICENSE_POLICY`` spec (default: the ModusQ §5.6
        ``default_license`` map). ``None`` for a settings-less ``open`` — the low-level seam applies no
        license-class filter (matching the cross-encoder's settings-less behavior)."""
        if settings is None:
            return None
        spec = settings.components.get(LICENSE_POLICY, _DEFAULT_LICENSE_POLICY)
        return ComponentFactory.get().create_as(spec, LicensePolicy)

    @classmethod
    async def open(
        cls,
        repository: DocumentRepository,
        embedder: Embedder,
        settings: Settings | None = None,
        *,
        llm: LanguageModel | None = None,
    ) -> RetrievalEngine:
        """Validate index compatibility, then return a query-ready engine — the ``Searcher`` + (only when
        a reranker is configured) the cross-encoder, built from ``Settings``. Refuses on a schema or
        embedding-fingerprint mismatch. ``llm`` (the bridge components' model) may be injected — a
        composition root shares its one LLM this way; otherwise one is built from ``Settings`` and owned
        (closed by :meth:`aclose`). The store + embedder passed in stay the caller's."""
        meta = await repository.index_meta()
        if not meta.get("schema_version"):
            raise RetrievalError("retrieval index has not been built yet (no index_meta)")
        conflict = IndexMeta.conflict(meta, embedder)
        if conflict:
            raise RetrievalError(conflict)
        # The cross-encoder's model loads lazily, so building it here is cheap; it stays unused unless the
        # configured pipeline has a reranker (a settings-less ``open`` has none). Same for the LLM — built
        # lazily (no key/network until ``complete``), unused unless a bridge component (multi_query /
        # llm_judge) calls it; the lean default pipeline never does.
        cross_encoder = OnnxCrossEncoder.create(settings.rerank) if settings is not None else None
        owns_llm = llm is None and settings is not None
        if owns_llm:
            llm = LanguageModel.create(settings.llm)
        return cls(
            repository, embedder, cls.build_searcher(settings), cross_encoder,
            cls.build_license_policy(settings), llm, owns_llm=owns_llm,
        )

    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        repository: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        llm: LanguageModel | None = None,
    ) -> RetrievalEngine:
        """Open a query-ready engine from ``Settings`` — connecting the repository (the same store
        ingestion writes) + the shared embedder, then validating via :meth:`open`. A composition root
        (``TarnRag``) may inject a pre-built ``repository`` + ``embedder`` (and ``llm``) to share them
        across engines; pass both or neither (mirrors ``IngestionEngine.create``). What this builds, it
        owns and :meth:`aclose` releases; what is injected stays the caller's."""
        settings = settings or get_settings()
        built = repository is None and embedder is None
        if built:
            repository = await DocumentRepository.create(settings.database, settings.EMBEDDING_DIMENSION)
            embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
        elif repository is None or embedder is None:
            raise ValueError("inject both repository and embedder, or neither")
        try:
            engine = await cls.open(repository, embedder, settings, llm=llm)
        except BaseException:
            if built:  # e.g. an index-identity mismatch: release what we just built before re-raising
                embedder.close()
                await repository.disconnect()
            raise
        engine.owns_repository = engine._owns_embedder = built
        return engine

    async def aclose(self) -> None:
        """Release what this engine built: its LLM client, its embedder, then (base) its store."""
        if self._owns_llm and self._llm is not None:
            await self._llm.aclose()
        if self._owns_embedder:
            self.embedder.close()
        await super().aclose()

    def context(self) -> RetrievalContext:
        """The ``(store, embedder, cross-encoder, license policy, llm)`` the pipeline runs against — also
        what an eval sweep scores *alternative* pipeline specs over (``TarnRag.retrieval_context``)."""
        return RetrievalContext(
            self.repository, self.embedder, self._cross_encoder, self._license_policy, self._llm
        )

    async def search(self, query: Query) -> list[RetrievalResult]:
        """Run the configured retrieval pipeline (retrieve [license/scope pre-filter] → fuse → hydrate →
        merge → rerank → top_k → assemble)."""
        return await self._pipeline.search(query, self.context())

    async def search_text(
        self, text: str, *, top_k: int = 8, dense_k: int = 50
    ) -> list[RetrievalResult]:
        """Convenience over :meth:`search`: build a :class:`Query` from a raw string."""
        return await self.search(Query(text=text, top_k=top_k, dense_k=dense_k))

    async def explain(self, query: Query) -> SearchTrace:
        """Like :meth:`search`, but returns the recorded :class:`SearchTrace` — the pipeline's intermediate
        stages (each retriever's candidates before fusion, the fused / merged / reranked / final rankings,
        and any routing decision) — instead of only the final results. The data behind ``TarnRag.explain``
        and the console's ``explain`` command; ``trace.results`` is the same ranking ``search`` returns."""
        trace = SearchTrace(query=query)
        await self._pipeline.search(query, self.context(), trace)
        return trace

    async def explain_text(self, text: str, *, top_k: int = 8, dense_k: int = 50) -> SearchTrace:
        """Convenience over :meth:`explain`: build a :class:`Query` from a raw string."""
        return await self.explain(Query(text=text, top_k=top_k, dense_k=dense_k))
