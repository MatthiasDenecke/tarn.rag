"""Run the multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue, distractor) through tarn.rag's
generation engine and print F1 / EM against MOTHRAG's published numbers.

Stream the datasets from HuggingFace with ``--hf`` (no download), and point the reader at any
OpenAI-compatible endpoint (e.g. a Llama-3.3-70B server, to match MOTHRAG's reader) via env::

    LLM__PROVIDER=openai \
    LLM__MODEL=meta-llama/Llama-3.3-70B-Instruct \
    LLM__API_BASE_URL=https://your-endpoint/v1 \
    LLM__API_KEY=sk-... \
    python scripts/run_benchmarks.py hotpotqa --hf --limit 200 --sweep

Start with a small ``--limit`` to gauge cost/quality (one multi-hop answer is several LLM calls). The
embedder + the GENERATION_PIPELINE (reasoner / grounding) are whatever Settings selects (env / .env). The
comparison is a baseline, not a controlled replication (different reader/embedder; lean reasoners).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import (
    BRIDGE_RETRIEVAL,
    HYBRID_RETRIEVAL,
    build_corpus_index,
    format_comparison,
    format_sweep,
    run_benchmark,
    run_over_corpus,
    sweep_benchmark,
    sweep_over_corpus,
)
from tarnrag.eval.benchmarks import HF_LOADERS, LOADERS, corpus_from_items


async def _run(
    dataset: str, path: str | None, limit: int | None, hf: bool, sweep: bool, bridge: bool,
    corpus: str, corpus_limit: int | None, concurrency: int, reasoners: list[str] | None, grounding: str | None,
    hybrid: bool, retrieval_spec: str | None,
) -> None:
    settings = get_settings()
    tag = ""
    if bridge:  # Phase-2 bridge retrieval (multi-query + LLM judge) instead of the lean dense-only default
        settings.components[RETRIEVAL_PIPELINE] = BRIDGE_RETRIEVAL
        tag = " + bridge"
    if hybrid:  # dense + sparse BM25, RRF-fused (overrides --bridge if both given)
        settings.components[RETRIEVAL_PIPELINE] = HYBRID_RETRIEVAL
        tag = " + hybrid"
    if retrieval_spec:  # an explicit pinned spec (e.g. the S1 routed profile) — overrides the flags above
        settings.components[RETRIEVAL_PIPELINE] = json.loads(retrieval_spec)
        tag = " + custom-retrieval"
    llm = LanguageModel.create(settings.llm)
    try:
        if corpus == "pool":
            await _run_over_pool(
                dataset, path, limit, hf, sweep, settings, llm, tag, corpus_limit, concurrency, reasoners,
                grounding,
            )
        else:
            await _run_distractor(dataset, path, limit, hf, sweep, settings, llm, tag, reasoners, grounding)
    finally:
        await llm.aclose()  # the reader's network client


async def _run_distractor(dataset, path, limit, hf, sweep, settings, llm, tag, reasoners, grounding) -> None:
    items = HF_LOADERS[dataset](limit=limit) if hf else LOADERS[dataset](path, limit=limit)
    print(
        f"running {len(items)} {dataset} questions (distractor) "
        f"through reader={settings.llm.provider}:{settings.llm.model}{tag} …"
    )
    if sweep:
        reports = await sweep_benchmark(items, llm, reasoners=reasoners, settings=settings, grounding=grounding)
        print(format_sweep(dataset, reports))
    else:
        print(format_comparison({dataset: await run_benchmark(items, llm, settings=settings)}))


async def _run_over_pool(
    dataset, path, limit, hf, sweep, settings, llm, tag, corpus_limit, concurrency, reasoners, grounding
) -> None:
    # The corpus is built from the first ``corpus_limit`` dev questions (all of them if None) — sized larger
    # than the eval slice so the eval questions' gold passages are in the haystack. A moderate corpus_limit
    # keeps the (per-doc) build tractable while still making retrieval miss-able.
    all_items = HF_LOADERS[dataset](limit=corpus_limit) if hf else LOADERS[dataset](path, limit=corpus_limit)
    corpus = corpus_from_items(all_items)
    eval_items = all_items[:limit] if limit else all_items
    # Tag the index by embedder (model + dim) so different embedders get separate corpora — the index is
    # embedder-specific (its vectors + fingerprint), so a gte-small build mustn't be reused for te3-small.
    emb_tag = f"{settings.embedding.model.split('/')[-1]}_{settings.EMBEDDING_DIMENSION}"
    db_path = f"./data/bench_{dataset}_pool_{corpus_limit or 'full'}_{emb_tag}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"building corpus index for {dataset}: {len(corpus)} passages -> {db_path} (cached) …")
    repo, embedder = await build_corpus_index(corpus, settings, db_path=db_path)
    try:
        print(
            f"running {len(eval_items)} {dataset} questions over the shared corpus "
            f"through reader={settings.llm.provider}:{settings.llm.model}{tag} …"
        )
        if sweep:
            reports = await sweep_over_corpus(
                eval_items, llm, repo, embedder, reasoners=reasoners, settings=settings,
                concurrency=concurrency, grounding=grounding,
            )
            print(format_sweep(dataset, reports))
        else:
            report = await run_over_corpus(
                eval_items, llm, repo, embedder, settings=settings, concurrency=concurrency
            )
            print(format_comparison({dataset: report}))
    finally:
        embedder.close()
        await repo.disconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a MOTHRAG benchmark through tarn.rag's generation engine")
    parser.add_argument("dataset", choices=sorted(LOADERS), help="which benchmark")
    parser.add_argument("path", nargs="?", help="path to the downloaded dataset file (omit with --hf)")
    parser.add_argument(
        "--hf", action="store_true", help=f"stream from HuggingFace instead of a file ({sorted(HF_LOADERS)})"
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument(
        "--sweep", action="store_true",
        help="sweep the reasoners (single_hop / iterative / decomposition) instead of one configured run",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="retrieve with dense + sparse BM25, RRF-fused (vs the dense-only default); needs the corpus's "
             "FTS index, which ingestion already builds",
    )
    parser.add_argument(
        "--bridge", action="store_true",
        help="use the Phase-2 bridge retrieval (multi-query expansion + LLM relevance judge) — needs an LLM",
    )
    parser.add_argument(
        "--retrieval-spec", default=None, metavar="JSON",
        help="pin an explicit RETRIEVAL_PIPELINE spec as JSON (e.g. the S1 routed profile) — "
             "overrides --hybrid/--bridge; a measurement never inherits a default",
    )
    parser.add_argument(
        "--corpus", choices=["distractor", "pool"], default="distractor",
        help="retrieval setting: 'distractor' (per-question pool, default) or 'pool' (one shared corpus "
             "built from the dev set — the fullwiki-style setting; built once + cached)",
    )
    parser.add_argument(
        "--corpus-limit", type=int, default=None,
        help="with --corpus pool: build the corpus from only the first N dev questions (a moderate haystack "
             "the per-doc ingest can build in reasonable time); defaults to the whole dev set",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="with --corpus pool: how many questions to answer in parallel (the LLM calls are I/O-bound, "
             "so this overlaps their latency; bounded by the API rate limit)",
    )
    parser.add_argument(
        "--reasoners", type=lambda s: [r.strip() for r in s.split(",") if r.strip()], default=None,
        help="with --sweep: comma-separated reasoner class_names to sweep (e.g. "
             "'decomposition,grounded_retrieval'); defaults to single_hop,iterative,decomposition",
    )
    parser.add_argument(
        "--grounding", default=None,
        help="grounding checker for the grounded_retrieval reasoner (e.g. llm_grounding); default heuristic",
    )
    args = parser.parse_args(argv)
    if args.hf and args.dataset not in HF_LOADERS:
        parser.error(f"--hf supports {sorted(HF_LOADERS)}; {args.dataset!r} needs a file path")
    if not args.hf and not args.path:
        parser.error("provide a dataset file path, or use --hf")
    asyncio.run(
        _run(
            args.dataset, args.path, args.limit, args.hf, args.sweep, args.bridge,
            args.corpus, args.corpus_limit, args.concurrency, args.reasoners, args.grounding,
            args.hybrid, args.retrieval_spec,
        )
    )


if __name__ == "__main__":
    main()
