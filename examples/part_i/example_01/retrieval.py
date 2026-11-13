"""Example 01 — Retrieval (embedded mode, SQLite).

Query the store that ``ingestion.py`` built. The same ``TarnRag`` facade opens that store and embeds
each query with the SAME embedding identity ingestion used (the config is shared via
``examples/common.py``), runs a dense k-NN search over the SQLite vector index, and returns ranked,
provenance-bearing hits.

Run it after the ingestion example has populated the store (same one-time setup — see
``ingestion.py``):

    python -m examples.part_i.example_01.retrieval
"""

from __future__ import annotations

import asyncio

from examples.common import base_settings, example_db, require_model
from tarnrag import RetrievalResult, TarnRag

QUERIES = [
    "How do I check a storage tank for corrosion?",
    "Which animal is native to Western Australia?",
]


async def main() -> dict[str, list[RetrievalResult]]:
    """Run each query and return {query: ranked hits} (also printed)."""
    require_model()
    db_path = example_db(__file__)  # the same store the ingestion example wrote
    if not db_path.exists():
        raise SystemExit(
            f"No store at {db_path}. Run the ingestion example first:\n"
            f"  python -m examples.part_i.example_01.ingestion"
        )

    settings = base_settings(db_path)

    # `async with TarnRag(...)` opens the same SQLite store (sharing the embedding identity from the
    # shared config). `retrieve` embeds the query and returns up to `top_k` hits, best first, wrapped
    # in an `Outcome` (`.value` is the hits; `.report` would carry any issues — none for a plain query).
    async with TarnRag(settings) as tarn:
        answers = {query: (await tarn.retrieve(query, top_k=3)).value for query in QUERIES}

    for query, results in answers.items():
        print(f"\nQuery: {query!r}")
        if not results:
            print("  (no results)")
        # RetrievalResult.score: higher is better; results arrive already ranked. The default
        # pipeline is hybrid (dense + sparse, RRF-fused), so the score is the RRF sum
        # `Σ 1/(60 + rank)`: ~0.033 = both retrievers ranked it 1st, ~0.016 = only one returned it.
        # Per-retriever raw scores stay in hit.component_scores.
        for rank, hit in enumerate(results, start=1):
            snippet = hit.text[:88].replace("\n", " ")
            print(f"  {rank}. score={hit.score:+.3f}  document={hit.document_id!r}")
            print(f"     {snippet}...")
    return answers


if __name__ == "__main__":
    asyncio.run(main())
