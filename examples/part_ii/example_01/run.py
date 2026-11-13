"""Part II · Example 01 — dense-only retrieval (where it shines, where it breaks).

Reuses the base store + config from Example 00 (a single ``dense`` retriever + identity fuser — no config
change yet) and runs two probes that show dense embeddings' strength and blind spot:

  * good — a paraphrase with no shared words: dense finds the right doc by *meaning* (a keyword search
    would miss it);
  * bad  — a bare opaque part number: there's no meaning to embed, so dense ranks the right doc out of the
    top-k. Example 02 (sparse + RRF) repairs it.

Run from the repo root::

    python -m examples.part_ii.example_01.run

…or exercise the same base config interactively::

    python -m tarnrag.console examples/part_ii/example_00/config.yaml
    tarn> explain service a rotary fluid machine before powering it up
    tarn> explain XQ-9920-A
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parents[1] / "example_00" / "config.yaml"  # the base config (dense)


async def main() -> tuple[bool | None, bool | None]:
    """Run the good + bad probes; return their HIT/MISS verdicts so a test can assert the failure is real."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 01 · dense-only retrieval",
            shows="dense embeddings match by MEANING — a paraphrase with no shared words still finds the doc",
            fails="a bare opaque identifier (a part number) has no meaning to embed, so dense ranks it out of reach",
            fixed_next="Example 02 adds a sparse (BM25) retriever for exact-term matching and fuses it with RRF",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        good = await runner.show_retrieval(
            "service a rotary fluid machine before powering it up",
            gold="mechanical seal", label="good · paraphrase (no shared words)", top_k=4,
        )
        bad = await runner.show_retrieval(
            "XQ-9920-A", gold="XQ-9920-A", label="bad · opaque part number", top_k=4,
        )
        return good, bad


if __name__ == "__main__":
    asyncio.run(main())
