"""Example 03 — Comparing retrieval methods with the eval harness (embedded mode, SQLite).

The payoff of the retrieval seams: **comparing methods is just swapping the RetrievalPipeline spec.**
This sweeps three pipelines — dense, sparse (BM25), and hybrid (RRF over both) — against the SAME index
that ``example_03/ingestion.py`` built, scores each over a small labeled query set (``evalset.json``),
and prints a hit@k / MRR / nDCG@k comparison table — then a **per-query-type** breakdown.

The query set is labeled ``semantic`` (paraphrases with little keyword overlap) vs ``lexical`` (the exact
terms from the docs). Segmenting the scores by that label answers the question that decides whether
*query routing* is worth building: **does a different method win on a different kind of query?** (Spoiler
for this corpus: sparse BM25 collapses on the semantic queries — no shared terms — while dense holds up.)

When it does, a **`RoutingRetrievalPipeline`** classifies each query and sends it to the per-type-best
method. The ``"routed (by type)"`` spec below does that; the eval supplies the labeled ``query_type``, so
it routes on those labels (an *oracle*, isolating "does routing help?" from classifier accuracy), and the
domain-independent **`StructuralQueryClassifier`** then reproduces those labels with no labels of its own
(the classifier check at the end), so the same routing happens automatically in production.

Routing's ceiling is the best method *on each type*, so it only beats a single pipeline when the per-type
winners differ. On this tiny corpus dense is robust on both types, so ``routed`` simply matches ``dense``;
the case where routing wins outright (each method best on one type) is in ``tests/eval/test_harness.py``.

The corpus is tiny, so treat the numbers as illustrative; the point is the machinery. Relevance is
content-based: a hit counts if a result's chunk text contains one of a query's gold phrases.

Run (after the ingestion example):

    python -m examples.part_i.example_03.ingestion
    python -m examples.part_i.example_03.evaluation
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import base_settings, example_db, require_model
from tarnrag import TarnRag
from tarnrag.eval import EvalReport, EvalSet, format_reports, format_segmented, sweep
from tarnrag.retrieval import Query, StructuralQueryClassifier

EVALSET = Path(__file__).resolve().parent / "evalset.json"

# Comparing retrieval methods = different RetrievalPipeline specs against the same index.
PIPELINES = {
    "dense": {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]},
    "sparse (bm25)": {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "sparse"}],
        "fuser": {"class_name": "identity"},
    },
    "hybrid (rrf)": {
        "class_name": "retrieval_pipeline",
        "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
        "fuser": {"class_name": "rrf"},
    },
}

# Routing: classify the query, then dispatch to the per-type-best method (sparse on lexical, dense on
# semantic). Because the eval supplies the labeled query_type, the router dispatches on those labels and
# skips its classifier entirely (a caller-supplied query_type wins) = oracle routing, which isolates "does
# routing help?" from classifier accuracy. The classifier check at the end shows the domain-independent
# StructuralQueryClassifier reproducing those labels with no labels of its own.
ROUTED = {
    "class_name": "routing_retrieval_pipeline",
    "routes": {"lexical": PIPELINES["sparse (bm25)"], "semantic": PIPELINES["dense"]},
    "default": PIPELINES["dense"],
}


async def main() -> dict[str, EvalReport]:
    """Sweep the pipelines over the labeled set against the example_03 store; print the table."""
    require_model()
    db_path = example_db(__file__)  # the store example_03/ingestion.py wrote
    if not db_path.exists():
        raise SystemExit(
            f"No store at {db_path}. Run the ingestion example first:\n"
            f"  python -m examples.part_i.example_03.ingestion"
        )
    settings = base_settings(db_path)
    evalset = EvalSet.from_json(EVALSET)

    # `TarnRag` opens the store; `retrieval_context()` hands back the shared (store + embedder) the
    # pipeline specs all run against — so the sweep can score *alternative* pipelines (not just the
    # one configured) over this single index.
    async with TarnRag(settings) as tarn:
        ctx = tarn.retrieval_context()
        reports = await sweep({**PIPELINES, "routed (by type)": ROUTED}, ctx, evalset, k=3)

    print(
        f"Comparing {len(reports)} retrieval pipelines over {len(evalset.queries)} queries "
        "(k=3, relevance = gold phrase in the chunk text):\n"
    )
    print(format_reports(reports))
    print("\nSegmented by query type — does a different method win on a different kind of query?\n")
    print(format_segmented(reports, metric="hit_at_k"))
    print()
    print(format_segmented(reports, metric="mrr"))

    # The router above used the labeled types (oracle). Show the domain-independent classifier
    # reproducing them with no labels — so that routing is what you'd get automatically in production.
    print("\nStructuralQueryClassifier (domain-independent, no labels) vs the labels:")
    classifier = StructuralQueryClassifier(StructuralQueryClassifier.Config())
    for q in evalset.queries:
        probe = Query(text=q.text)
        classifier.classify(probe, None)
        flag = "ok  " if probe.query_type == q.query_type else "MISS"
        print(f"  [{flag}] predicted {probe.query_type:9} labeled {q.query_type:9} {q.text}")
    return reports


if __name__ == "__main__":
    asyncio.run(main())
