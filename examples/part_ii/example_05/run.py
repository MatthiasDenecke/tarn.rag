"""Part II · Example 05 — query routing (Act A capstone).

Act A's capstone steps back and *compares* the retrieval methods over a small labeled set, then routes.
The point: a different method wins on a different *kind* of query — dense on paraphrases (semantic), sparse
on exact terms / identifiers (lexical) — so a `routing_retrieval_pipeline` that classifies each query and
dispatches it to the per-type-best method beats every fixed pipeline. A `StructuralQueryClassifier`
reproduces the labels with none of its own, so the routing happens automatically in production.

Run from the repo root (after Example 00 built the base store)::

    python -m examples.part_ii.example_05.run
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag
from tarnrag.core.engine.config import RETRIEVAL_PIPELINE
from tarnrag.eval import EvalSet, format_reports, format_segmented, sweep
from tarnrag.retrieval import Query, StructuralQueryClassifier

CONFIG = Path(__file__).resolve().parent / "config.yaml"
EVALSET = Path(__file__).resolve().parent / "evalset.yaml"

# The three fixed baselines the scoreboard compares against the routed pipeline.
_DENSE = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]}
_SPARSE = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "sparse"}],
           "fuser": {"class_name": "identity"}}
_HYBRID = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
           "fuser": {"class_name": "rrf"}}


async def main() -> tuple[dict, int]:
    """Return (the sweep reports, the classifier's correct-label count) so a test can assert the lesson."""
    require_model()
    settings = load_config(CONFIG)
    routed = settings.components[RETRIEVAL_PIPELINE]  # the routing pipeline this example configures
    evalset = EvalSet.from_records(yaml.safe_load(EVALSET.read_text()))
    async with TarnRag(settings) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 05 · query routing (Act A capstone)",
            shows="a different method wins on a different kind of query (dense on paraphrases, sparse on "
            "exact terms) — so routing each query to the per-type best beats any fixed pipeline",
            fixed_next="Act B — generation: turn retrieved passages into a grounded answer (Example 06)",
        )
        await runner.ensure_corpus(corpus("corpus-2"))

        # 1) the scoreboard — dense / sparse / hybrid / routed, segmented by query type.
        #    The eval supplies each query's type, so `routed` dispatches on those labels (oracle routing):
        #    it isolates "does routing help?" from classifier accuracy (measured separately, below).
        reports = await sweep(
            {"dense": _DENSE, "sparse": _SPARSE, "hybrid": _HYBRID, "routed": routed},
            tarn.retrieval_context(), evalset, k=3,
        )
        print("\n" + format_reports(reports))
        print("\n" + format_segmented(reports, metric="hit_at_k"))

        # 2) the classifier reproduces those labels with none of its own — so routing is automatic.
        classifier = StructuralQueryClassifier(StructuralQueryClassifier.Config())
        correct = 0
        print("\nStructuralQueryClassifier (no labels of its own) vs the gold labels:")
        for q in evalset.queries:
            probe = Query(text=q.text)
            classifier.classify(probe, None)
            ok = probe.query_type == q.query_type
            correct += ok
            print(f"  [{'ok ' if ok else 'MISS'}] predicted {probe.query_type:9} labeled {q.query_type:9} {q.text}")

        # 3) routing in action — the explain breakdown's `routed:` line shows which route each query took.
        await runner.show_retrieval("XQ-9920-A", label="lexical → sparse", top_k=3)
        await runner.show_retrieval(
            "how do I keep a big metal container from rusting over time", label="semantic → dense", top_k=3,
        )
        return reports, correct


if __name__ == "__main__":
    asyncio.run(main())
