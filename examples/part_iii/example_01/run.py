"""Part III · Example 01 — a conversation WITHOUT memory.

Two turns, the second leaning on the first::

    you › Which frame is the TX-200 built on?
    you › Which oil grade does it require?

With ``conversation_memory.enabled: false`` every question stands alone, so the follow-up is searched
exactly as typed. Nothing in "Which oil grade does it require?" says *which* frame — and this store
holds five near-identical frame lubrication specs. Retrieval returns three of the others; the one that
answers the question is never put in front of the reader.

That MISS is printed by a check that involves no model at all. Example 02 flips one config line and
plays the same conversation again.

Run from the repo root (offline — a scripted model is injected, no API key needed)::

    python -m examples.part_iii.example_01.run
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import load_config, require_model
from examples.part_iii._conversation import TURNS, Narrator, Played, scripted_model
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> tuple[list[Played], int]:
    """Play the conversation; return each turn's outcome and how many turns were remembered (the smoke
    test asserts the follow-up missed and that nothing was remembered)."""
    require_model()
    settings = load_config(CONFIG)

    async with TarnRag(settings, llm=scripted_model()) as tarn:
        narrator = Narrator(tarn, settings)
        narrator.banner(
            "Part III · Example 01 · a conversation without memory",
            shows="every question is searched exactly as typed — fine for the first question of a "
            "conversation, which stands on its own",
            fails="the follow-up. 'Which oil grade does it require?' never says which frame, so "
            "retrieval returns OTHER frames' oil specs and the right one never reaches the reader",
            fixed_next="Example 02 turns conversation memory on (one config line): the follow-up is "
            "rewritten into a standalone question before it is searched",
        )
        await narrator.ensure_corpus()
        played = [await narrator.play(turn) for turn in TURNS]
        remembered = await narrator.show_history()
    return played, remembered


if __name__ == "__main__":
    asyncio.run(main())
