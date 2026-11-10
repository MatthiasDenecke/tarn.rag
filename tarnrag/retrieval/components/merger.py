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


class Merger(Component):
    """Port: rewrite the ranked result set (e.g. collapse sibling leaves into their section parent)."""

    class Config(Component.Config):
        """Base merger config; concrete mergers pin ``class_name``."""

    @abstractmethod
    async def merge(
        self, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        """Return a rewritten, still best-first result list."""


class AutoMerger(Merger):
    """Auto-merging: when ``min_siblings`` or more retrieved leaves share a ``parent_chunk_id``, replace
    them with their section parent (hydrated on demand). The parent takes its best child's rank slot and
    ``score`` (the ``max`` over the merged children); ``component_scores`` are the per-retriever max.

    The parent is assumed to share its children's license / availability — the license-scope filter has
    already dropped any disallowed leaves — but a parent that is *itself* unavailable is skipped (its
    leaves stay). Strict per-purpose re-filtering of the parent is a deferred refinement.
    """

    class Config(Merger.Config):
        class_name: Literal["auto_merge"] = "auto_merge"
        min_siblings: int = Field(default=2, ge=2)  # collapse only when ≥ this many siblings are retrieved

    config: AutoMerger.Config

    async def merge(
        self, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        # Group the result positions by the parent they descend from (leaves only carry a parent id).
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(results):
            parent_id = r.provenance.parent_chunk_id if r.provenance else None
            if parent_id:
                groups.setdefault(parent_id, []).append(i)
        to_merge = {pid: idxs for pid, idxs in groups.items() if len(idxs) >= self.config.min_siblings}
        if not to_merge:
            return results

        parents = {rec.chunk_id: rec for rec in await ctx.store.hydrate(list(to_merge))}
        merged_at: dict[int, RetrievalResult] = {}  # anchor (best child) position -> the parent result
        dropped: set[int] = set()
        for pid, idxs in to_merge.items():
            parent = parents.get(pid)
            if parent is None or not parent.available:  # parent gone / not retrievable -> keep the leaves
                continue
            children = [results[i] for i in idxs]
            components: dict[str, float] = {}
            for c in children:
                for name, s in c.component_scores.items():
                    components[name] = max(components.get(name, s), s)
            merged_at[min(idxs)] = RetrievalResult.from_record(
                parent, score=max(c.score for c in children), component_scores=components
            )
            dropped.update(idxs)

        # Rebuild in rank order: the parent takes its best child's slot; the other children drop out;
        # everything else passes through. Dedupe by chunk_id (a parent that was also retrieved directly).
        out: list[RetrievalResult] = []
        seen: set[str] = set()
        for i, r in enumerate(results):
            hit = merged_at.get(i) or (None if i in dropped else r)
            if hit is None or hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
        return out
