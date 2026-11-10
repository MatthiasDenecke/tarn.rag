"""The RoutingRetrievalPipeline — classify a query, dispatch to the per-type-best sub-pipeline.

A ``RoutingRetrievalPipeline`` is a ``Searcher`` (so the engine builds it from the ``RETRIEVAL_PIPELINE``
spec in place of a plain pipeline). It holds a ``QueryClassifier`` + a ``{query_type → Searcher}`` route
map + a ``default`` ``Searcher``. On ``search``: if the query isn't already classified, run the classifier
to populate ``query_type`` (+ annotations) on a *copy* — the caller's ``Query`` is never mutated; then
dispatch to ``routes.get(query_type, default)``. Routes are themselves ``Searcher``s (built recursively), so
a route is just another pipeline spec — sparse for ``lexical``, dense for ``semantic``, a nested router, … —
and the route map is the deployment's config, tuned from the segmented eval.

Nesting: a label this router's classifier produced is *its* route key, not the next router's — so a nested
router receives the query with ``query_type`` cleared (annotations kept) and runs its own classifier. A
caller-supplied ``query_type`` is the caller's routing decision and passes through unchanged at every level.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from bausatz import ComponentFactory
from pydantic import Field

from tarnrag.contracts import RetrievalResult
from tarnrag.retrieval.components.classifier import (
    GenericQueryClassifier,
    QueryClassifier,
)
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.pipeline.pipeline import RetrievalPipeline
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query, SearchTrace


class RoutingRetrievalPipeline(Searcher):
    """Classify a query and dispatch it to the route matching its ``query_type`` (else the default)."""

    class Config(Searcher.Config):
        class_name: Literal["routing_retrieval_pipeline"] = "routing_retrieval_pipeline"
        classifier: QueryClassifier.Config = Field(default_factory=GenericQueryClassifier.Config)
        routes: dict[str, Searcher.Config] = Field(default_factory=dict)  # query_type → Searcher
        # the fallthrough route: a bare retrieval_pipeline (dense + identity fuser)
        default: Searcher.Config = Field(default_factory=RetrievalPipeline.Config)

    config: RoutingRetrievalPipeline.Config

    def __init__(self, config: RoutingRetrievalPipeline.Config) -> None:
        super().__init__(config)
        self._classifier: QueryClassifier | None = None
        self._routes: dict[str, Searcher] = {}
        self._default: Searcher | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the classifier + each route + the default through the factory (routes are ``Searcher``s,
        so routing composes — a route may be a plain pipeline or a nested router)."""
        self._classifier = factory.build_as(self.config.classifier, QueryClassifier)
        self._routes = {t: factory.build_as(cfg, Searcher) for t, cfg in self.config.routes.items()}
        self._default = factory.build_as(self.config.default, Searcher)

    async def search(
        self, query: Query, ctx: RetrievalContext, trace: SearchTrace | None = None
    ) -> list[RetrievalResult]:
        self.ensure_built()
        supplied = bool(query.query_type)  # a caller-supplied query_type wins; otherwise classify
        query = replace(query, annotations=list(query.annotations))  # classify a copy, never the caller's
        if not supplied:
            self._classifier.classify(query, ctx)
        searcher = self._routes.get(query.query_type, self._default)
        if trace is not None:  # record which route this query took before delegating to it
            route_key = query.query_type if query.query_type in self._routes else "default"
            trace.routing = (query.query_type, route_key)
        if not supplied and isinstance(searcher, RoutingRetrievalPipeline):
            query = replace(query, query_type="")  # a nested router classifies by its own taxonomy
        return await searcher.search(query, ctx, trace)
