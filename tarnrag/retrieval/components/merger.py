"""The Merger component family: post-retrieval auto-merging.

A ``Merger`` is an optional, config-driven ``Component`` that rewrites the surviving (already filtered)
result set before truncation. ``auto_merge`` collapses retrieved sibling leaf chunks into their shared
section parent (via the persisted ``parent_chunk_id``), so the model gets one coherent section instead of
several adjacent fragments. Like the other retrieval seams it's a pure strategy — the store is injected at
call time via the ``RetrievalContext`` (the parent chunk is hydrated on demand).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from bausatz import Component
from pydantic import Field

from tarnrag.contracts import RetrievalResult
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class Merger(Component):
    """Port: rewrite the ranked result set (e.g. collapse sibling leaves into their section parent)."""

    class Config(Component.Config):
        """Base merger config; concrete mergers pin ``class_name``."""

    @abstractmethod
    async def merge(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        """Return a rewritten, still best-first result list. ``query`` is the one being answered — any
        chunk the merger brings in (rather than receives) must pass ``ctx.filter_for(query)``, exactly as
        the retrievers' candidates did."""


class AutoMerger(Merger):
    """Auto-merging over the multi-level section tree: when ``min_siblings`` or more results share a
    ``parent_chunk_id``, replace them with their section parent (hydrated on demand); repeat until no
    group qualifies, so merged subsections can merge again into *their* parent. A parent takes its best
    child's rank slot and ``score`` (the ``max`` over the merged children); ``component_scores`` are the
    per-retriever max.

    A parent enters the results only if it passes the query's permitted-chunk filter
    (``ctx.filter_for(query)`` — availability, ``ai_grounding_allowed`` under ``GENERATION_GROUNDING``,
    license class, and method scope); otherwise its children stay. The retrievers pre-filter in SQL, but a
    parent is hydrated here, outside that path, and need not share its leaves' scope (a method bundle may
    hold leaves without their section). Finally a result whose text is already covered by a returned
    ancestor (same document, higher level, a superset of its char spans) is dropped, so a section and one
    of its own fragments never both reach the reader.
    """

    class Config(Merger.Config):
        class_name: Literal["auto_merge"] = "auto_merge"
        min_siblings: int = Field(default=2, ge=2)  # collapse only when ≥ this many siblings are retrieved

    config: AutoMerger.Config

    async def merge(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        permitted = ctx.filter_for(query)
        refused: set[str] = set()  # parents that are missing or not permitted — never retried
        merged = list(results)
        while True:  # each productive pass shrinks the list (≥ 2 children → 1 parent), so this terminates
            groups = self._sibling_groups(merged, refused)
            if not groups:
                break
            parents = {rec.chunk_id: rec for rec in await ctx.store.hydrate(list(groups))}
            merged_at: dict[int, RetrievalResult] = {}  # anchor (best child) position -> the parent
            dropped: set[int] = set()
            for pid, idxs in groups.items():
                parent = parents.get(pid)
                if parent is None or not permitted.admits(parent):  # gone / not for this query: keep leaves
                    refused.add(pid)
                    continue
                children = [merged[i] for i in idxs]
                components: dict[str, float] = {}
                for c in children:
                    for name, s in c.component_scores.items():
                        components[name] = max(components.get(name, s), s)
                merged_at[min(idxs)] = RetrievalResult.from_record(
                    parent, score=max(c.score for c in children), component_scores=components
                )
                dropped.update(idxs)
            if not merged_at:
                break
            merged = self._rebuild(merged, merged_at, dropped)
        return self._drop_covered(merged)

    def _sibling_groups(
        self, results: list[RetrievalResult], refused: set[str]
    ) -> dict[str, list[int]]:
        """Result positions grouped by their parent chunk id — only groups big enough to merge, and never
        a parent already refused."""
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(results):
            parent_id = r.provenance.parent_chunk_id if r.provenance else None
            if parent_id and parent_id not in refused:
                groups.setdefault(parent_id, []).append(i)
        return {pid: idxs for pid, idxs in groups.items() if len(idxs) >= self.config.min_siblings}

    @staticmethod
    def _rebuild(
        results: list[RetrievalResult], merged_at: dict[int, RetrievalResult], dropped: set[int]
    ) -> list[RetrievalResult]:
        """Rebuild in rank order: a parent takes its best child's slot; its other children drop out;
        everything else passes through. Dedupe by chunk_id (a parent that was also retrieved directly)."""
        out: list[RetrievalResult] = []
        seen: set[str] = set()
        for i, r in enumerate(results):
            hit = merged_at.get(i) or (None if i in dropped else r)
            if hit is None or hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
        return out

    @staticmethod
    def _drop_covered(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Drop each result whose content a higher-level result of the same document already carries.
        A section parent's geometry is the concatenation of its descendants' char spans (the chunker's
        aggregation; geometry is persisted, so it survives hydration), so containment is a subset test on
        the ``(start, end)`` footprints — no store round-trip. Retrievers return only leaves (the stores'
        ``level = 0`` predicate), so every section here is one this merger built: the case is a leaf left
        behind under a parent merged at a higher level. (Pieces of one oversize element share its span but
        sit at the same level, so neither covers the other.)"""
        def footprint(r: RetrievalResult) -> set[tuple[int, int]]:
            return {(s.start, s.end) for s in r.provenance.geometry} if r.provenance else set()

        sections = [(s, footprint(s)) for s in results if s.provenance and s.provenance.level > 0]
        if not sections:
            return results

        def covered(r: RetrievalResult) -> bool:
            mine = footprint(r)
            return bool(mine) and any(
                s.document_id == r.document_id and s.provenance.level > r.provenance.level and mine <= theirs
                for s, theirs in sections
            )

        return [r for r in results if not covered(r)]
