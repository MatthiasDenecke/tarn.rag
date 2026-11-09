"""RetrievalPipeline — the retrieval composition as a container Component.

A *sibling* of the ingestion ``Pipeline`` (both are config-driven container ``Component``s; not a
subclass — retrieval's flow is heterogeneous: parallel retrievers + fan-in + fixed steps). It builds the
configured retrievers + fuser (+ optional merger / reranker) as children and owns the flow:

    retrieve (parallel, over-fetch) → fuse → hydrate → filter → auto-merge → rerank → top_k → assemble.

The ``RetrievalEngine`` is a thin facade over it: it does the compatibility check + store/embedder/
cross-encoder construction, then delegates.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field

from tarnrag.contracts import RetrievalResult
from bausatz import ComponentFactory
from tarnrag.retrieval.components.fuser import Fuser, IdentityFuser
from tarnrag.retrieval.components.merger import Merger
from tarnrag.retrieval.components.reranker import Reranker
from tarnrag.retrieval.components.retriever import DenseRetriever, RetrievalContext, Retriever
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query, SearchTrace


class RetrievalPipeline(Searcher):
    """Compose the configured retrievers + fuser and run the retrieval flow over a ``RetrievalContext``."""

    class Config(Searcher.Config):
        class_name: Literal["retrieval_pipeline"] = "retrieval_pipeline"
        # Typed children: each spec validates into its family's Config (a foreign child — e.g. a fuser
        # in the retrievers list — is rejected when the pipeline spec is validated).
        retrievers: list[Retriever.Config] = Field(default_factory=lambda: [DenseRetriever.Config()])
        fuser: Fuser.Config = Field(default_factory=IdentityFuser.Config)
        merger: Merger.Config | None = None  # optional auto-merging step (none ⇒ skipped)
        reranker: Reranker.Config | None = None  # optional second-pass rerank (none ⇒ skipped)

    config: RetrievalPipeline.Config

    def __init__(self, config: RetrievalPipeline.Config) -> None:
        super().__init__(config)
        self._retrievers: list[Retriever] = []
        self._fuser: Fuser | None = None
        self._merger: Merger | None = None
        self._reranker: Reranker | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the retriever + fuser (+ optional merger / reranker) children through the factory."""
        self._retrievers = [factory.build_as(cfg, Retriever) for cfg in self.config.retrievers]
        self._fuser = factory.build_as(self.config.fuser, Fuser)
        self._merger = factory.build_as(self.config.merger, Merger) if self.config.merger else None
        self._reranker = (
            factory.build_as(self.config.reranker, Reranker) if self.config.reranker else None
        )

    async def search(
        self, query: Query, ctx: RetrievalContext, trace: SearchTrace | None = None
    ) -> list[RetrievalResult]:
        self.ensure_built()
        lists = await asyncio.gather(*(r.retrieve(query, ctx) for r in self._retrievers))
        keys = self._retriever_keys()
        per_retriever = dict(zip(keys, lists))
        if trace is not None:
            trace.record_retrievers(keys, lists)
        # Candidates are already license/scope-filtered inside each retriever (over-fetched), so the
        # fused set is permitted; merge / rerank, then top_k.
        fused = self._fuser.fuse(per_retriever)
        records = {rec.chunk_id: rec for rec in await ctx.store.hydrate([h.chunk_id for h in fused])}
        results = [
            RetrievalResult.from_record(rec, score=h.score, component_scores=h.component_scores)
            for h in fused
            if (rec := records.get(h.chunk_id)) is not None
        ]
        if trace is not None:
            trace.record_stage("fused", results)
        if self._merger is not None:
            results = await self._merger.merge(query, results, ctx)
            if trace is not None:
                trace.record_stage("merged", results)
        if self._reranker is not None:
            results = await self._reranker.rerank(query, results, ctx)
            if trace is not None:
                trace.record_stage("reranked", results)
        final = results[: query.top_k]
        if trace is not None:
            trace.record_stage("final", final)
        return final

    def _retriever_keys(self) -> list[str]:
        """A distinct key per retriever (for the per-retriever candidate map + the fused
        ``component_scores``): its configured ``name``, else its ``class_name``. Duplicates are
        disambiguated with a ``#n`` suffix, so two same-class retrievers don't collide on one key and
        silently drop one's candidates — the common (uniquely-named / single-class) case is unchanged."""
        keys: list[str] = []
        counts: dict[str, int] = {}
        for r in self._retrievers:
            base = r.config.name or r.config.class_name
            n = counts.get(base, 0)
            counts[base] = n + 1
            keys.append(base if n == 0 else f"{base}#{n}")
        return keys
