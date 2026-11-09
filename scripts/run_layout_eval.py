"""Layout-aware retrieval eval on TAT-QA — source-hit@k, dense vs hybrid, segmented by table / text.

    EMBEDDING__MODEL=thenlper/gte-small EMBEDDING__MODEL_DIR=./models/gte-small \
    python scripts/run_layout_eval.py --limit 100

Builds one shared corpus of every TAT-QA table + paragraph (cached), then for each extractive question
measures whether the top-k retrieved chunks include the gold answer-source element — overall and split by
``answer_from`` (table vs text). Runs dense and hybrid (dense + BM25, RRF) back to back so the table-vs-text
gap, and whether hybrid closes it, are visible in one shot.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import HYBRID_RETRIEVAL
from tarnrag.eval.layout import (
    SourceHitReport,
    build_tatqa_index,
    format_numeric,
    load_tatqa_numeric,
    tatqa_numeric,
    format_attribution,
    format_source_hit,
    load_tatqa,
    stream_tatqa,
    tatqa_attribution,
    tatqa_source_hit,
)
from tarnrag.retrieval.components.classifier import StructuralQueryClassifier
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


# ---------------------------------------------------------------------------- the S1 sweep (--s1)
# Query understanding as the shipped default (doc/s1-scope.md): sweep the candidate quality-profile
# pieces over one cached index, then compose the routed profile from the measured winners. Every
# leg's spec is pinned fully — a measurement never inherits a default.

_CE_RERANKER = {"class_name": "cross_encoder", "top_n": 20}  # the P4 winner, pinned incl. top_n


def _weighted_hybrid(sparse_weight: float) -> dict:
    """The sparse-weighted hybrid (S1 PR-2): classic hybrid with the sparse arm up-weighted."""
    return {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
        "fuser": {"class_name": "rrf", "weights": {"sparse": sparse_weight}},
    }


def _llm_hybrid(dense_arm: str) -> dict:
    """Hybrid with the dense arm replaced by an LLM-assisted bridge retriever (multi_query | hyde)."""
    return {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": dense_arm}, {"class_name": "sparse"}],
        "fuser": {"class_name": "rrf"},
    }


def _structural_labels(queries, repo, embedder) -> list[str]:
    """Label every question with the structural classifier — the same deterministic labels the routed
    leg dispatches on, so the slices below measure exactly what each route would receive."""
    classifier = StructuralQueryClassifier(StructuralQueryClassifier.Config())
    ctx = RetrievalContext(store=repo, embedder=embedder)
    labels: list[str] = []
    for q in queries:
        query = Query(text=q.question)
        classifier.classify(query, ctx)
        labels.append(query.query_type)
    return labels


def _slice_rates(hits: list[bool], labels: list[str]) -> dict[str, tuple[int, float]]:
    """Hit-rate per structural label (n + rate), from the report's per-query hits."""
    rates: dict[str, tuple[int, float]] = {}
    for label in sorted(set(labels)):
        seg = [h for h, current in zip(hits, labels) if current == label]
        rates[label] = (len(seg), sum(seg) / len(seg)) if seg else (0, 0.0)
    return rates


def _slice_rate(report: SourceHitReport, labels: list[str], label: str) -> tuple[int, float]:
    return _slice_rates(report.hits, labels).get(label, (0, 0.0))


def _format_slices(rates: dict[str, tuple[int, float]]) -> str:
    parts = [f"{label}: {rate:.3f} (n={n})" for label, (n, rate) in rates.items()]
    return "structural slice   " + "   ".join(parts)


def _pick_route(
    candidates: list[str], results: dict[str, SourceHitReport], labels: list[str], slice_label: str
) -> str:
    """The route winner: best source-hit on its slice, ties to the earliest candidate (list the
    cheapest first). An empty slice yields no evidence — the first (baseline) candidate wins."""
    n, _ = _slice_rate(results[candidates[0]], labels, slice_label)
    if n == 0:
        return candidates[0]
    return max(candidates, key=lambda name: _slice_rate(results[name], labels, slice_label)[1])


async def _s1_sweep(queries, repo, embedder, settings, *, k: int, concurrency: int) -> None:
    labels = _structural_labels(queries, repo, embedder)
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"structural labels over {len(labels)} queries: {counts}")
    has_llm = bool(settings.llm.resolved_api_key())
    if not has_llm:
        print("no LLM key resolved -> skipping the multi_query / hyde legs "
              f"(set {settings.llm.key_env_var()} and LLM__PROVIDER/LLM__MODEL)")

    specs: dict[str, dict] = {
        "HYBRID": dict(HYBRID_RETRIEVAL),
        "HYBRID+CE": {**HYBRID_RETRIEVAL, "reranker": dict(_CE_RERANKER)},
        "W-SPARSE-1.5": _weighted_hybrid(1.5),
        "W-SPARSE-2": _weighted_hybrid(2.0),
        "W-SPARSE-3": _weighted_hybrid(3.0),
    }
    if has_llm:
        specs["MQ+SPARSE"] = _llm_hybrid("multi_query")
        specs["HYDE+SPARSE"] = _llm_hybrid("hyde")

    results: dict[str, SourceHitReport] = {}

    async def _run_leg(name: str, spec: dict) -> None:
        settings.components[RETRIEVAL_PIPELINE] = spec
        report = await tatqa_source_hit(
            queries, repo, embedder, settings=settings, k=k, concurrency=concurrency
        )
        results[name] = report
        print("\n" + format_source_hit(report, tag=f"[{name}]"))
        print(_format_slices(_slice_rates(report.hits, labels)))

    for name, spec in specs.items():
        await _run_leg(name, spec)

    # Compose the routed profile from the measured winners (baseline first — ties go to the cheaper leg).
    lexical_winner = _pick_route(
        ["HYBRID", "W-SPARSE-1.5", "W-SPARSE-2", "W-SPARSE-3"], results, labels, "lexical"
    )
    semantic_candidates = ["HYBRID"] + (["MQ+SPARSE", "HYDE+SPARSE"] if has_llm else [])
    semantic_winner = _pick_route(semantic_candidates, results, labels, "semantic")
    print(f"\nrouted composition: lexical -> {lexical_winner}, semantic -> {semantic_winner}, "
          "default -> HYBRID")
    routes = {"lexical": dict(specs[lexical_winner]), "semantic": dict(specs[semantic_winner])}
    routed = {
        "class_name": "routing_retrieval_pipeline",
        "classifier": {"class_name": "structural"},
        "routes": routes,
        "default": dict(HYBRID_RETRIEVAL),
    }
    await _run_leg("ROUTED", routed)
    routed_ce = {
        **routed,
        "routes": {t: {**spec, "reranker": dict(_CE_RERANKER)} for t, spec in routes.items()},
        "default": {**HYBRID_RETRIEVAL, "reranker": dict(_CE_RERANKER)},
    }
    await _run_leg("ROUTED+CE", routed_ce)

    print("\n=== S1 sweep summary (source-hit@%d, n=%d) ===" % (k, len(queries)))
    segs = sorted({seg for r in results.values() for seg in r.by_segment})
    header = f"{'leg':<14}" + "".join(f"{s:>12}" for s in segs) + f"{'lexical':>12}{'semantic':>12}{'overall':>12}"
    print(header + "\n" + "-" * len(header))
    for name, r in results.items():
        row = f"{name:<14}" + "".join(f"{r.by_segment.get(s, (0, 0.0))[1]:>12.3f}" for s in segs)
        lex_n, lex = _slice_rate(r, labels, "lexical")
        sem_n, sem = _slice_rate(r, labels, "semantic")
        print(row + f"{lex:>12.3f}{sem:>12.3f}{r.source_hit:>12.3f}")


async def _run(
    limit: int | None, k: int, concurrency: int, attribution: bool, hybrid: bool, table_view: str,
    rerank: list[str], numeric: bool, s1: bool,
) -> None:
    settings = get_settings()
    rows = stream_tatqa(limit)
    corpus, queries = load_tatqa(rows, limit=limit)
    emb_tag = f"{settings.embedding.model.split('/')[-1]}_{settings.EMBEDDING_DIMENSION}"
    if settings.embedding.contextualize_tables:  # a distinct index — the embed text differs (P1)
        emb_tag += "_ctx"
    db_path = f"./data/tatqa_{limit or 'full'}_{emb_tag}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"building TAT-QA corpus: {len(corpus)} elements -> {db_path} (cached) … ; {len(queries)} extractive queries")
    repo, embedder = await build_tatqa_index(corpus, settings, db_path=db_path)
    try:
        # Pin the retrieval spec EXPLICITLY either way — a measurement never inherits a default.
        dense_only = {
            "class_name": "retrieval_pipeline",
            "retrievers": [{"class_name": "dense"}],
            "fuser": {"class_name": "identity"},
        }
        settings.components[RETRIEVAL_PIPELINE] = HYBRID_RETRIEVAL if hybrid else dense_only
        if s1:  # the S1 sweep: candidate quality-profile legs + the routed composition
            await _s1_sweep(queries, repo, embedder, settings, k=k, concurrency=concurrency)
        elif numeric:  # PP-6: the arithmetic slice through table_lookup — deterministic, LLM-free
            nq = load_tatqa_numeric(rows, limit=limit)
            print(f"scoring {len(nq)} arithmetic questions through table_lookup (no LLM) …")
            spec = {"class_name": "table_lookup", "fallback": None, "top_k": k, "row_coverage": 0.5}
            report = await tatqa_numeric(
                nq, repo, embedder, settings=settings, reasoner_spec=spec, k=k, concurrency=concurrency
            )
            print("\n" + format_numeric(report, tag="[HYBRID]" if hybrid else "[DENSE]"))
        elif attribution:  # generation + LLM-judged citations (needs an LLM); retrieval per `hybrid`
            llm = LanguageModel.create(settings.llm)
            print(f"scoring attribution through reader={settings.llm.provider}:{settings.llm.model}"
                  f"{' + hybrid' if hybrid else ''}, table_view={table_view} …")
            try:
                overall, by_seg = await tatqa_attribution(
                    queries, llm, repo, embedder, settings=settings, table_view=table_view,
                    concurrency=concurrency,
                )
            finally:
                await llm.aclose()  # the reader's network client
            print("\n" + format_attribution(overall, by_seg))
        else:  # retrieval-only source-hit: dense vs hybrid (vs reranked), each leg's spec pinned explicitly.
            legs = [("DENSE", dense_only), ("HYBRID", HYBRID_RETRIEVAL)]
            for rr in rerank:  # P4: a second-pass reranker on top of the hybrid shortlist.
                # The spec is pinned FULLY (top_n included) — a measurement never inherits a default.
                legs.append((
                    f"HYBRID+{rr.upper()}",
                    {**HYBRID_RETRIEVAL, "reranker": {"class_name": rr, "top_n": 20}},
                ))
            for label, spec in legs:
                settings.components[RETRIEVAL_PIPELINE] = spec
                report = await tatqa_source_hit(queries, repo, embedder, settings=settings, k=k, concurrency=concurrency)
                print("\n" + format_source_hit(report, tag=f"[{label}]"))
    finally:
        embedder.close()
        await repo.disconnect()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TAT-QA layout-aware eval: source-hit@k (dense vs hybrid) or attribution")
    p.add_argument("--limit", type=int, default=100, help="cap TAT-QA records (corpus + queries); default 100")
    p.add_argument("-k", type=int, default=10, help="top-k retrieved chunks counted for source-hit")
    p.add_argument("--concurrency", type=int, default=8, help="parallel retrievals / answers")
    p.add_argument(
        "--attribution", action="store_true",
        help="instead of source-hit: answer each question and have an LLM judge the citations "
             "(grounded_rate = attribution precision) + F1/EM, segmented by table/text — needs an LLM",
    )
    p.add_argument("--hybrid", action="store_true", help="retrieve with dense + BM25 (RRF) instead of dense-only")
    p.add_argument(
        "--table-view", choices=["structured", "text"], default=None,
        help="how the reader + grounding judge see table chunks (REQUIRED with --attribution; "
             "pinned explicitly — a measurement never inherits a default)",
    )
    p.add_argument(
        "--rerank", action="append", choices=["cross_encoder", "llm_judge"], default=[],
        help="add a HYBRID+<reranker> source-hit leg (repeatable; spec pinned explicitly per leg). "
             "cross_encoder needs the local ONNX model; llm_judge needs an LLM key.",
    )
    p.add_argument("--numeric", action="store_true",
                   help="PP-6: run the arithmetic slice through the table_lookup reasoner (no LLM)")
    p.add_argument("--s1", action="store_true",
                   help="S1: sweep the quality-profile legs (hybrid ± CE, weighted-sparse, multi_query, "
                        "hyde) + the routed composition; LLM legs need a key (else skipped)")
    args = p.parse_args(argv)
    if args.attribution and args.table_view is None:
        p.error("--attribution requires --table-view (pin the view; don't inherit a default)")
    asyncio.run(_run(
        args.limit, args.k, args.concurrency, args.attribution, args.hybrid,
        args.table_view or "structured", args.rerank, args.numeric, args.s1,
    ))


if __name__ == "__main__":
    main()
