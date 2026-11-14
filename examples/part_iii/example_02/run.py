"""Part III · Example 02 — the same conversation WITH memory.

Example 01 searched the follow-up as typed and missed. One config line changes here —
``conversation_memory.enabled: true`` — and the same two turns are played again::

    you › Which frame is the TX-200 built on?
    you › Which oil grade does it require?
          ↳ searched: Which oil grade does the Cooper-Bessemer GMV frame require?

Before anything else runs, the follow-up is rewritten against the conversation into a question that
stands on its own. Retrieval, the reasoner and the proof tree then work on that question exactly as
they always did — none of them knows a conversation exists. The earlier turn itself is never shown to
the reader: memory decides *which question is asked*, never *what counts as evidence*.

Run from the repo root (offline — a scripted model is injected, no API key needed)::

    python -m examples.part_iii.example_02.run
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
    test asserts the follow-up was rewritten, hit, and that both turns were remembered)."""
    require_model()
    settings = load_config(CONFIG)

    async with TarnRag(settings, llm=scripted_model()) as tarn:
        narrator = Narrator(tarn, settings)
        narrator.banner(
            "Part III · Example 02 · the same conversation with memory",
            shows="the follow-up is rewritten against the last few turns into a standalone question "
            "BEFORE it is searched — so the right frame's oil spec is retrieved, and answered from",
            fails="nothing here — but note the cost: one extra LLM call per follow-up (none on a "
            "conversation's first turn), and a bad rewrite would mislead the whole turn",
        )
        await narrator.ensure_corpus()
        played = [await narrator.play(turn) for turn in TURNS]
        remembered = await narrator.show_history()
    return played, remembered


if __name__ == "__main__":
    asyncio.run(main())
