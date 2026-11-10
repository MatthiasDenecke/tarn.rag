"""Example 02 — Retrieval over the JSON-configured store (embedded mode, SQLite).

Queries the store that ``example_02/ingestion.py`` built with the small-chunk pipeline. Note what
this script does NOT need: the pipeline JSON. Retrieval only embeds the query and searches — it
depends on the *embedding identity* (shared via ``base_settings``), not on how ingestion chunked.
So this is the same ``base_settings(db_path)`` as example 01; the smaller chunks just mean shorter,
more focused snippets come back.

Run (after the ingestion example has populated the store):

    python -m examples.part_i.example_02.retrieval
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
    """Run each query against the small-chunk store; return {query: ranked hits} (also printed)."""
    require_model()
    db_path = example_db(__file__)  # the same store example_02's ingestion wrote
    if not db_path.exists():
        raise SystemExit(
            f"No store at {db_path}. Run the ingestion example first:\n"
            f"  python -m examples.part_i.example_02.ingestion"
        )

    settings = base_settings(db_path)  # no pipeline JSON needed — retrieval only needs the embedding

    async with TarnRag(settings) as tarn:
        answers = {query: (await tarn.retrieve(query, top_k=3)).value for query in QUERIES}

    for query, results in answers.items():
        print(f"\nQuery: {query!r}")
        if not results:
            print("  (no results)")
        for rank, hit in enumerate(results, start=1):
            snippet = hit.text[:88].replace("\n", " ")
            print(f"  {rank}. score={hit.score:+.3f}  document={hit.document_id!r}")
            print(f"     {snippet}...")
    return answers


if __name__ == "__main__":
    asyncio.run(main())
