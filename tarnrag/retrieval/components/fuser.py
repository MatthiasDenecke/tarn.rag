"""The Fuser component family: merge per-retriever candidate lists into one ranked list.

A ``Fuser`` is a config-driven ``Component``. ``identity`` is the single-retriever passthrough (the flow
stays uniform — always retrieve → fuse); ``rrf`` is Reciprocal Rank Fusion for hybrid retrieval. Each
``FusedHit`` keeps the per-retriever scores (``component_scores``) so the breakdown stays transparent.

Both fusers return their hits best-first with a **deterministic tie-break** — by ``score`` descending,
then ``chunk_id`` ascending (the ModusQ §5.5 / §9 contract). The tie-break is what makes identical inputs
yield byte-identical orderings across runs and ports; without it, equal-score hits would fall back to
dict / iteration order, which the future C++ port could not reproduce.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Literal

from bausatz import Component
from pydantic import Field

from tarnrag.contracts import Candidate


@dataclass
class FusedHit:
    """A fused hit: a chunk id, its fused ``score`` (higher = better), and the per-retriever scores."""

    chunk_id: str
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)


def _ranked(hits: list[FusedHit]) -> list[FusedHit]:
    """Order fused hits best-first with the mandatory deterministic tie-break: ``score`` descending, then
    ``chunk_id`` ascending (ModusQ §5.5 / §9). Equal-score hits are broken by id, not by input order, so
    identical inputs yield identical orderings across runs and ports. The single home for the ordering
    contract every ``Fuser`` returns."""
    return sorted(hits, key=lambda h: (-h.score, h.chunk_id))


class Fuser(Component):
    """Port: merge ``{retriever_name: candidates}`` into one ranked list of ``FusedHit``s (best first)."""

    class Config(Component.Config):
        """Base fuser config; concrete fusers pin ``class_name``."""

    @abstractmethod
    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        """Fuse the per-retriever candidate lists into one ranked list."""


class IdentityFuser(Fuser):
    """Single-retriever passthrough: keep the retriever's ranking (``score = -rank``, so higher is
    better and ``top_k`` truncation by score works), surfacing its raw score as the component score."""

    class Config(Fuser.Config):
        class_name: Literal["identity"] = "identity"

    config: IdentityFuser.Config

    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        return _ranked(
            [
                FusedHit(c.chunk_id, score=-float(c.rank), component_scores={name: c.raw_score})
                for name, candidates in per_retriever.items()
                for c in candidates
            ]
        )


class RRFFuser(Fuser):
    """Reciprocal Rank Fusion: ``score(chunk) = Σ_retrievers weight_r / (k + rank)`` — rank-based, so it
    fuses retrievers with incomparable raw scores (cosine distance vs BM25). Optional per-retriever
    ``weights`` tilt the fusion (S1: the sparse-weighted hybrid for lexical queries) without touching
    the rank-based core; the empty default is classic unweighted RRF."""

    class Config(Fuser.Config):
        class_name: Literal["rrf"] = "rrf"
        k: int = Field(default=60, gt=0)
        # Per-retriever-key weight. Keys are the pipeline's retriever keys (configured ``name``, else
        # ``class_name``, ``#n``-deduped — ``RetrievalPipeline._retriever_keys``); unlisted keys weigh
        # 1.0, so the empty default keeps every existing spec's fusion byte-identical. Only the ratios
        # matter: uniform scaling scales every fused score alike and never changes the ranking, so
        # {sparse: 2} ≡ {sparse: 4, dense: 2} — don't normalize, compare weight configs by ratio.
        weights: dict[str, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)

    config: RRFFuser.Config

    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for name, candidates in per_retriever.items():
            weight = self.config.weights.get(name, 1.0)
            for c in candidates:
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + weight / (self.config.k + c.rank)
                components.setdefault(c.chunk_id, {})[name] = c.raw_score
        return _ranked(
            [FusedHit(cid, score=s, component_scores=components[cid]) for cid, s in scores.items()]
        )
