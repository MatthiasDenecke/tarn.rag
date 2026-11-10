"""Retrieval eval harness — compare retrieval-pipeline specs (and embed-time variants) on a labeled set.

``sweep`` scores many ``RetrievalPipeline`` specs against one index (the method comparison); embed-time
variants are compared by sweeping against a separately-built index. Relevance is content-based — a result
is relevant iff its text contains a gold phrase — so labels survive re-ingestion (see ``dataset``).
"""

from tarnrag.eval import metrics
from tarnrag.eval.benchmark_runner import (
    BRIDGE_RETRIEVAL,
    MOTHRAG_PUBLISHED,
    build_corpus_index,
    format_comparison,
    format_sweep,
    run_benchmark,
    run_over_corpus,
    sweep_benchmark,
    sweep_over_corpus,
)
from tarnrag.eval.benchmarks import (
    HF_LOADERS,
    LOADERS,
    BenchItem,
    corpus_from_items,
    load_2wiki,
    load_2wiki_hf,
    load_hotpotqa,
    load_hotpotqa_hf,
    load_musique,
    load_musique_hf,
)
from tarnrag.eval.dataset import EvalQuery, EvalSet
from tarnrag.eval.generation import (
    GenEvalQuery,
    GenEvalReport,
    GenEvalSet,
    GenQueryReport,
    citation_coverage,
    content_hit,
    evaluate_generation,
    exact_match,
    format_generation_reports,
    sweep_generation,
    token_f1,
)
from tarnrag.eval.harness import (
    EvalReport,
    QueryReport,
    by_query_type,
    evaluate_pipeline,
    format_reports,
    format_segmented,
    sweep,
)

__all__ = [
    "BRIDGE_RETRIEVAL",
    "HF_LOADERS",
    "LOADERS",
    "MOTHRAG_PUBLISHED",
    # MOTHRAG benchmark harness
    "BenchItem",
    # retrieval eval
    "EvalQuery",
    "EvalReport",
    "EvalSet",
    # generation eval
    "GenEvalQuery",
    "GenEvalReport",
    "GenEvalSet",
    "GenQueryReport",
    "QueryReport",
    "build_corpus_index",
    "by_query_type",
    "citation_coverage",
    "content_hit",
    "corpus_from_items",
    "evaluate_generation",
    "evaluate_pipeline",
    "exact_match",
    "format_comparison",
    "format_generation_reports",
    "format_reports",
    "format_segmented",
    "format_sweep",
    "load_2wiki",
    "load_2wiki_hf",
    "load_hotpotqa",
    "load_hotpotqa_hf",
    "load_musique",
    "load_musique_hf",
    "metrics",
    "run_benchmark",
    "run_over_corpus",
    "sweep",
    "sweep_benchmark",
    "sweep_generation",
    "sweep_over_corpus",
    "token_f1",
]
