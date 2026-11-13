"""Part II · Example 08 — multi-hop reasoning (decomposition).

Example 07 verified its answers. But ``single_hop`` can only answer from what the QUESTION itself
retrieves in ONE pass — and some answers are reachable only through a *bridge*:

    compressor-models : "The TX-200 is ... built on a Cooper-Bessemer GMV frame"
    lubrication-spec  : "Cooper-Bessemer GMV frames require ... ISO VG 68"

Neither document holds the whole chain. And the second one is **not retrievable from the question**:
this example's store also holds four near-identical frame specs (Ariel JGT → VG 100, Waukesha VHP →
VG 46, Superior MH → VG 150, Clark HRA → VG 32), so "which oil grade does the TX-200 require?" pulls
back *those* — the GMV spec is crowded out of the top-k. Only something that knows to search for "GMV
frame" — i.e. a second hop, informed by the first — can find it.

So the failure is a *reachability* failure, not a reading failure: single_hop honestly answers "not
specified", because the fact it needs was never put in front of it. ``decomposition`` splits the
question into sub-questions, retrieves for each, and synthesizes — so the second hop lands the bridge.

This example has its OWN store (``multihop.db``): it ingests corpus-2 **plus** ``corpus-2-frames``,
leaving the base store the other rungs share untouched.

Run from the repo root::

    python -m examples.part_ii.example_08.run

`ask` needs the LLM key in OPENAI_LLM_KEY (loaded from the repo-root .env) and `pip install '.[openai]'`.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

from tarnrag import TarnRag
from tarnrag.core.engine.config import GENERATION_PIPELINE

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner

CONFIG = Path(__file__).resolve().parent / "config.yaml"

QUESTION = "Which oil grade does the TX-200 require?"
GOLD = "ISO VG 68"  # only reachable via the bridge: TX-200 → GMV frame → ISO VG 68

# The PREVIOUS rung's reasoner, run over this same store — the bad case.
SINGLE_HOP = {"class_name": "single_hop", "top_k": 4}


async def main() -> tuple[bool, bool]:
    """Return (decomposition answered with the gold grade, single_hop answered with it) — the test
    asserts the first is True and the second is False (multi-hop fixes what one hop cannot reach)."""
    require_model()
    settings = load_config(CONFIG)

    async with TarnRag(settings) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 08 · multi-hop reasoning (decomposition)",
            shows="a question whose answer spans TWO documents (TX-200 → GMV frame → ISO VG 68). "
            "decomposition retrieves per sub-question, so the second hop finds the bridge passage "
            "the question alone cannot reach",
            fails="nothing here — but note the cost: decomposition pays several LLM calls (decompose, "
            "retrieve per sub-question, synthesize) where single_hop paid one",
            fixed_next="Example 09 adds the answerability gate — refuse BEFORE the read when the "
            "evidence can't support the query, instead of paying for it first",
        )
        # This example's own store: corpus-2 + the four distractor frame specs.
        await runner.ensure_corpus(corpus("corpus-2"), corpus("corpus-2-frames"))

        # 1. MODEL-FREE, and the crux: the bridge passage is not in reach of the question at all.
        # No reasoner can read a passage that retrieval never returns.
        await runner.show_retrieval(
            QUESTION, gold=GOLD, top_k=4,
            label="out of reach · the GMV spec is crowded out by four near-identical frame specs",
        )

        # 2. the good case — this example's config (decomposition).
        multi = await runner.show_answer(
            QUESTION, gold=GOLD, label="multi-hop · decomposition bridges the two documents",
        )

    # 3. the bad case — the PREVIOUS rung's reasoner over the SAME store. It answers honestly
    # ("not specified"): the fact it needed was never retrieved. Note it is still reported
    # grounded and does NOT abstain — a non-answer makes no claims, so there is nothing to refute.
    single_settings = copy.deepcopy(settings)
    generation = dict(single_settings.components[GENERATION_PIPELINE])
    generation["reasoner"] = SINGLE_HOP
    single_settings.components[GENERATION_PIPELINE] = generation

    async with TarnRag(single_settings) as tarn:
        single = await Runner(tarn).show_answer(
            QUESTION, gold=GOLD, label="single hop (Example 07's reasoner) · cannot reach the bridge",
        )

    return (GOLD.lower() in multi.answer.lower()), (GOLD.lower() in single.answer.lower())


if __name__ == "__main__":
    asyncio.run(main())
