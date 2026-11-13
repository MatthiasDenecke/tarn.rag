"""Part II · Example 02 — hybrid retrieval (dense + sparse + RRF).

One config change from Example 01: the ``retrieval_pipeline`` now runs a dense AND a sparse (BM25)
retriever and fuses them with RRF. Watch the two probes from Example 01 swap places:

  * the opaque part number ``XQ-9920-A`` that dense MISSED now HITS (sparse's exact match, fused in);
  * but the pure paraphrase that dense NAILED now MISSES — RRF blends in sparse's ranking, and sparse
    boosts surface-word matches, demoting the semantic answer. Fusion is a trade-off, not a free lunch.

Example 03 (a cross-encoder reranker) recovers the paraphrase while keeping the part-number win.

Run from the repo root (after Example 00 built the store)::

    python -m examples.part_ii.example_02.run

…or interactively::

    python -m tarnrag.console examples/part_ii/example_02/config.yaml
    tarn> explain XQ-9920-A
    tarn> explain service a rotary fluid machine before powering it up
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> tuple[bool | None, bool | None]:
    """Run the two probes; return their HIT/MISS verdicts so a test can assert the swap is real."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 02 · hybrid retrieval (dense + sparse + RRF)",
            shows="adding a sparse (BM25) retriever + RRF fixes the opaque part number dense missed",
            fails="but sparse noise demotes the pure-paraphrase query dense nailed — fusion is a trade-off",
            fixed_next="Example 03 adds a cross-encoder reranker that recovers the paraphrase, keeping both",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        fixed = await runner.show_retrieval(
            "XQ-9920-A", gold="XQ-9920-A", label="fixed · opaque part number (was a dense miss)", top_k=4,
        )
        regressed = await runner.show_retrieval(
            "service a rotary fluid machine before powering it up",
            gold="mechanical seal", label="new miss · paraphrase (sparse noise demotes it)", top_k=4,
        )
        return fixed, regressed


if __name__ == "__main__":
    asyncio.run(main())
