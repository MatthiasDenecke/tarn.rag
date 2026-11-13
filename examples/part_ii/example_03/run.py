"""Part II · Example 03 — cross-encoder reranking.

One config change from Example 02: a cross-encoder ``reranker`` on the ``retrieval_pipeline``. It re-scores
the fused top candidates by true query–passage relevance, so it gets the **best of both** — it recovers the
paraphrase that hybrid demoted AND keeps the part-number win:

  * paraphrase   → dense ✓ → hybrid ✗ → rerank ✓
  * part number  → dense ✗ → hybrid ✓ → rerank ✓

But the reranker ranks individual *chunks*, so a question whose answer spans a whole section comes back as
a fragment — the single best chunk holds only some of the steps. Example 04 (structure-aware chunking +
auto-merge) returns the whole coherent section.

Run from the repo root::

    python -m examples.part_ii.example_03.run

Needs the reranker model (a separate one-time fetch)::

    python scripts/fetch_model.py --repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> tuple[bool | None, bool | None, bool | None]:
    """Run the three probes; return their HIT/MISS verdicts so a test can assert the lesson."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 03 · cross-encoder reranking",
            shows="the reranker re-scores the fused candidates by true relevance — recovering the paraphrase "
            "hybrid demoted AND keeping the part-number win (best of both)",
            fails="but it ranks individual chunks, so a section-spanning question returns a fragment — the "
            "single best chunk has only some of the steps",
            fixed_next="Example 04 re-ingests with structure-aware chunking + auto-merge → the whole section",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        recovered = await runner.show_retrieval(
            "service a rotary fluid machine before powering it up",
            gold="mechanical seal", label="recovered · paraphrase (hybrid had demoted it)", top_k=4,
        )
        kept = await runner.show_retrieval(
            "XQ-9920-A", gold="XQ-9920-A", label="kept · opaque part number", top_k=4,
        )
        fragmented = await runner.show_retrieval(
            "what are all the steps in the reciprocating compressor startup procedure?",
            gold="loading sequence", label="new miss · the top chunk is only a fragment of the section",
            top_k=1,
        )
        return recovered, kept, fragmented


if __name__ == "__main__":
    asyncio.run(main())
