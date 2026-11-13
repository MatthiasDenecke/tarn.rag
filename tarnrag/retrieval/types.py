"""Retrieval query types (ModusQ spec §5.1). Result types live in ``tarnrag.contracts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tarnrag.contracts import (
    Annotation,
    Candidate,
    ChunkFilter,
    MethodRef,
    RetrievalResult,
)


class Purpose(str, Enum):
    """
    Retrieval intent (ModusQ §5.1) — drives scope/licensing policy in later steps.
    """

    EXECUTION = "EXECUTION"
    AUTHORING = "AUTHORING"
    GENERATION_GROUNDING = "GENERATION_GROUNDING"


# Sentinel for "no scope restriction" (whole index).
ALL = "ALL"


@dataclass
class Query:
    """
    A retrieval request: the query text plus knobs (purpose, method scope, top_k / dense_k /
    sparse_k), and a classification (``query_type`` + ``annotations``) a ``QueryClassifier`` may fill.

    ``query_type`` is the cheap route key the ``RoutingRetrievalPipeline`` dispatches on; ``annotations``
    is the rich, extensible channel a classifier writes its findings to — the same ``Annotation`` type
    enrichment uses on chunks (so an LLM classifier's findings carry the ``deterministic`` flag, and a
    span can mark which substring is an identifier). Both default empty: an unclassified query routes to
    the pipeline's default. Set them via a classifier or supply them directly.
    """

    text: str
    purpose: Purpose = Purpose.EXECUTION
    scope: list[MethodRef] | str = ALL  # MethodRef[] or ALL
    top_k: int = 8
    dense_k: int = 50
    sparse_k: int = 50  # candidate window for the sparse (BM25 / tsvector) retriever
    query_type: str = ""  # headline classification label; the router's route key
    annotations: list[Annotation] = field(default_factory=list)  # the classifier's rich findings
    # The results will ground an LLM's output (generation sets this; the caller's ``purpose`` is kept).
    # ADDITIVE, never a replacement for the purpose: it requires ``ai_grounding_allowed`` on top, and a
    # license policy intersects the purpose's classes with the ``GENERATION_GROUNDING`` ones — so grounding
    # can only tighten what the caller's own purpose may see, never widen it.
    grounds_generation: bool = False

    def permitted_filter(self) -> ChunkFilter:
        """The permitted-chunk filter for this query (ModusQ §5.6) — available-only by default, grounding
        required for ``GENERATION_GROUNDING`` or a query that ``grounds_generation``, restricted to ``scope``
        unless it is ``ALL``. The retrievers pass it into the store so disallowed chunks are dropped before
        ``top_k`` (with over-fetch), instead of truncating first and filtering after (which could
        under-return for a tight scope)."""
        return ChunkFilter(
            require_grounding=self.grounds_generation or self.purpose == Purpose.GENERATION_GROUNDING,
            method_scope=self._scope_refs(),
        )

    def _scope_refs(self) -> tuple[MethodRef, ...] | None:
        """The method scope as a tuple of refs, or ``None`` for ``ALL`` (the whole index) — normalizes the
        ``list[MethodRef] | str`` sentinel union in one place, so callers don't branch on the ALL string."""
        return None if self.scope == ALL else tuple(self.scope)


# ---------------- the explain trace (the data behind RetrievalEngine.explain / TarnRag.explain) ----------


@dataclass
class RetrieverCandidates:
    """One retriever's ranked candidates *before* fusion: its key (the configured ``name`` or the
    ``class_name``) and the raw, engine-specific hits (distance for dense KNN, BM25 for sparse)."""

    key: str
    candidates: list[Candidate]


@dataclass
class SearchStage:
    """A named snapshot of the ranked results at one point in the pipeline — ``fused`` → optional
    ``merged`` → optional ``reranked`` → ``final``. Diffing the chunk-id order of two consecutive stages
    is how a UI shows what merging or reranking *moved*."""

    name: str
    results: list[RetrievalResult]


@dataclass
class SearchTrace:
    """The recorded inner workings of one retrieval — what ``RetrievalEngine.explain`` / ``TarnRag.explain``
    return. Output-free data a UI renders: the per-retriever candidate lists before fusion, the ranked
    results at every pipeline stage, and the routing decision when a ``RoutingRetrievalPipeline`` dispatched.

    A ``Searcher`` populates it when it is passed as ``search``'s ``trace`` argument (``None`` ⇒ no tracing,
    the hot path). ``results`` is the final ranking — identical to what ``retrieve`` would return."""

    query: Query
    per_retriever: list[RetrieverCandidates] = field(default_factory=list)
    stages: list[SearchStage] = field(default_factory=list)
    # Each router's dispatch decision, outermost first — ``(query_type, route_key)`` per router passed
    # through; a nested router appends its own, so the whole path is visible, not just the last hop.
    route_path: list[tuple[str, str]] = field(default_factory=list)

    @property
    def routing(self) -> tuple[str, str] | None:
        """The decision of the router that picked the pipeline which actually ran (the innermost one);
        ``None`` when no router dispatched. See ``route_path`` for every hop."""
        return self.route_path[-1] if self.route_path else None

    def record_retrievers(self, keys: list[str], candidate_lists: list[list[Candidate]]) -> None:
        """Snapshot each retriever's pre-fusion candidates (paired key → candidate list)."""
        self.per_retriever = [
            RetrieverCandidates(key, list(cands)) for key, cands in zip(keys, candidate_lists)
        ]

    def record_stage(self, name: str, results: list[RetrievalResult]) -> None:
        """Snapshot the ranked results at a named stage (a copy, so a later stage doesn't mutate it)."""
        self.stages.append(SearchStage(name, list(results)))

    @property
    def results(self) -> list[RetrievalResult]:
        """The final ranked results (the last recorded stage) — what ``retrieve`` would return."""
        return self.stages[-1].results if self.stages else []
