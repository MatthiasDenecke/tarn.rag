"""The retrieval eval harness — run a labeled query set through retrieval pipelines and score them.

The unit of comparison is a ``RetrievalPipeline`` over a shared ``RetrievalContext`` (the index +
embedder + optional cross-encoder): **comparing retrieval methods = different pipeline specs against the
same index** (``sweep``). Embed-time variants (e.g. header-path injection) are compared by running the
sweep against a *different* context — an index rebuilt with the variant — since those change what's
indexed, not the query-time pipeline. Relevance is content-based (see ``dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bausatz import ComponentFactory

from tarnrag.eval.dataset import EvalSet
from tarnrag.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query


@dataclass
class QueryReport:
    """Per-query outcome: the ranked relevance flags + this query's metric values + its type label."""

    text: str
    relevances: list[bool]
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    query_type: str = ""


@dataclass
class EvalReport:
    """A pipeline's aggregate metrics (means over the query set) + the per-query breakdown."""

    k: int
    n: int
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: list[QueryReport] = field(default_factory=list)


def _is_relevant(text: str, phrases: list[str]) -> bool:
    """Content-based relevance: the result text contains one of the gold phrases (case-insensitive)."""
    low = text.lower()
    return any(phrase.lower() in low for phrase in phrases)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(per_query: list[QueryReport], k: int) -> EvalReport:
    """Mean the per-query metrics into an ``EvalReport`` — the one aggregation, used by both
    ``evaluate_pipeline`` (over the whole set) and ``by_query_type`` (over a type's slice)."""
    return EvalReport(
        k=k,
        n=len(per_query),
        hit_at_k=_mean([r.hit_at_k for r in per_query]),
        mrr=_mean([r.reciprocal_rank for r in per_query]),
        ndcg_at_k=_mean([r.ndcg_at_k for r in per_query]),
        per_query=per_query,
    )


async def evaluate_pipeline(
    pipeline: Searcher, ctx: RetrievalContext, evalset: EvalSet, *, k: int = 8
) -> EvalReport:
    """Run every query through ``pipeline`` (``top_k=k``) and score content-based relevance @k. The
    query's ``query_type`` label is carried onto the live ``Query``, so a routing pipeline dispatches on
    it (routing on the labeled types) while the metrics are still segmented by that same label."""
    per_query: list[QueryReport] = []
    for q in evalset.queries:
        query = Query(text=q.text, purpose=q.purpose, top_k=k, query_type=q.query_type)
        results = await pipeline.search(query, ctx)
        rels = [_is_relevant(r.text, q.relevant) for r in results]
        per_query.append(
            QueryReport(
                text=q.text,
                relevances=rels,
                hit_at_k=hit_at_k(rels, k),
                reciprocal_rank=reciprocal_rank(rels),
                ndcg_at_k=ndcg_at_k(rels, k),
                query_type=q.query_type,
            )
        )
    return _aggregate(per_query, k)


def by_query_type(report: EvalReport) -> dict[str, EvalReport]:
    """Re-aggregate a report's per-query results by ``query_type`` — the segmentation that reveals which
    method wins on which kind of query (the evidence for whether query routing would help). Types are
    returned sorted; an unlabeled set collapses to a single ``""`` group."""
    groups: dict[str, list[QueryReport]] = {}
    for q in report.per_query:
        groups.setdefault(q.query_type, []).append(q)
    return {qtype: _aggregate(qs, report.k) for qtype, qs in sorted(groups.items())}


async def sweep(
    specs: dict[str, dict[str, Any]], ctx: RetrievalContext, evalset: EvalSet, *, k: int = 8
) -> dict[str, EvalReport]:
    """Evaluate many named specs against the same context — the method comparison. Each spec is built
    into a ``Searcher`` (a ``RetrievalPipeline`` or a ``RoutingRetrievalPipeline``) and scored over the
    query set, so a router can be swept alongside the single pipelines it routes between."""
    factory = ComponentFactory.get()
    return {
        name: await evaluate_pipeline(factory.create_as(spec, Searcher), ctx, evalset, k=k)
        for name, spec in specs.items()
    }


def format_reports(reports: dict[str, EvalReport]) -> str:
    """Render a sweep's reports as a comparison table — one row per pipeline spec."""
    header = f"{'pipeline':<24}{'hit@k':>9}{'mrr':>9}{'ndcg@k':>9}"
    lines = [header, "-" * len(header)]
    for name, r in reports.items():
        lines.append(f"{name:<24}{r.hit_at_k:>9.3f}{r.mrr:>9.3f}{r.ndcg_at_k:>9.3f}")
    if reports:
        any_report = next(iter(reports.values()))
        lines.append(f"\n(n={any_report.n} queries, k={any_report.k})")
    return "\n".join(lines)


_METRIC_LABELS = {"hit_at_k": "hit@k", "mrr": "mrr", "ndcg_at_k": "ndcg@k"}


def format_segmented(reports: dict[str, EvalReport], metric: str = "mrr") -> str:
    """A **pipeline × query-type** table of one ``metric`` (``hit_at_k`` | ``mrr`` | ``ndcg_at_k``),
    with an ``all`` column for the overall mean — the per-type view that shows whether different query
    types favour different methods (and so whether query-type routing would beat any single pipeline)."""
    if metric not in _METRIC_LABELS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRIC_LABELS)}")
    segmented = {name: by_query_type(r) for name, r in reports.items()}
    types = sorted({qtype for seg in segmented.values() for qtype in seg})
    cols = [*types, "all"]
    w = max(9, max((len(c) for c in cols), default=0) + 2)
    header = f"{_METRIC_LABELS[metric] + ' by type':<24}" + "".join(f"{c:>{w}}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, report in reports.items():
        seg = segmented[name]
        values = [getattr(seg[t], metric) for t in types] + [getattr(report, metric)]
        lines.append(f"{name:<24}" + "".join(f"{v:>{w}.3f}" for v in values))
    if reports:
        any_seg = next(iter(segmented.values()))
        counts = ", ".join(f"{t}={any_seg[t].n}" for t in types)
        lines.append(f"\n(n per type: {counts}; k={next(iter(reports.values())).k})")
    return "\n".join(lines)
