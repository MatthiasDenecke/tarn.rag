"""Part II · Example 06 — minimal generation (the Act B opener).

Act A retrieved passages; Act B reads them into an ANSWER. The base generation_pipeline (single_hop reasoner
+ provenance assembler) does one retrieve→read pass and returns the answer plus a *proof tree* — which chunk
backed it. The good case: an answerable question comes back grounded, with its evidence.

The limitation this opener exposes (and Example 07 addresses): the minimal pipeline *trusts the reader* —
there's no grounding verification and no abstain. So a question the corpus can't answer comes back as a
terse non-answer that is still reported as grounded and non-abstained: the system can't tell "I answered"
from "I couldn't".

Run from the repo root (after Example 00 built the base store)::

    python -m examples.part_ii.example_06.run

`ask` needs the LLM key in OPENAI_LLM_KEY (loaded from the repo-root .env) and `pip install '.[openai]'`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> tuple[bool, bool]:
    """Return (the good answer is grounded in the gold fact, the minimal pipeline abstained on the
    unanswerable one) — the test asserts the first is True and the second is False (the limitation)."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 06 · minimal generation (Act B opener)",
            shows="single_hop reads the retrieved passages into a grounded answer + a proof tree — which "
            "chunk backed it",
            fails="but it trusts the reader: no grounding check, no abstain — a question the corpus can't "
            "answer returns a terse non-answer still marked grounded",
            fixed_next="Example 07 adds a grounding check + abstain — verify each claim, refuse when unsupported",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        good = await runner.show_answer(
            "How do I service a centrifugal pump before starting it?",
            gold="mechanical seal", label="grounded · answerable from the corpus",
        )
        # The corpus describes the cartridge mechanical seal but states no temperature rating — single_hop
        # honestly returns a non-answer ("not provided"), yet the minimal pipeline still reports it as
        # grounded and does NOT abstain. That missing signal is what Example 07 adds.
        unanswerable = await runner.show_answer(
            "What is the maximum allowable working temperature of the cartridge mechanical seal?",
            label="limitation · not in the corpus, yet not flagged",
        )
        runner.note(f"unanswerable probe → grounded={unanswerable.grounded}, abstained={unanswerable.abstained}")
        return ("mechanical seal" in good.answer.lower()), unanswerable.abstained


if __name__ == "__main__":
    asyncio.run(main())
