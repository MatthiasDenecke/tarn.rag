"""Part II · Example 00 — ingest the shared corpus (build the `base` store).

Ingests ``examples/docs/corpus-2`` through the fully-explicit ``config.yaml``, then takes a first peek at
the result with one ``explain`` so you can see the machinery the rest of the ladder builds on. Run from the
repo root::

    python -m examples.part_ii.example_00.run

Needs the embedding model (``python scripts/fetch_model.py`` once). The same config is the console's hero
path — ``python -m tarnrag.console examples/part_ii/example_00/config.yaml`` — then ``ingest`` / ``explain``
interactively.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag
from tarnrag.ingestion.engine.types import DocumentSummary

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> list[DocumentSummary]:
    """Build the base store and return the per-document summary (so a test can assert the chunking)."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 00 · ingest the shared corpus",
            shows="the ingestion DAG — each .md file → markdown extractor → clean → recursive chunks → "
            "embeddings — written to one SQLite store that every later example queries",
            fixed_next="Example 01 queries this store with dense-only retrieval",
        )
        await runner.ingest_corpus(corpus("corpus-2"))
        # A first look at the store + the explain machinery (no good/bad framing yet — that's Example 01).
        await runner.show_retrieval("how do I prevent a pump from cavitating?", top_k=5)
        return (await tarn.docs()).value


if __name__ == "__main__":
    asyncio.run(main())
