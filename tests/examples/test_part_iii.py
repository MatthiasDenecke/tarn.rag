"""Smoke tests for Part III — the same two-turn conversation without (01) and with (02) memory.

Assert the lesson, not the wording: with memory OFF the follow-up is searched as typed and the passage
holding its answer never reaches the reader; with memory ON the follow-up is rewritten first and that
passage ranks first. Both examples inject the scripted offline model, so — unlike Part II's generation
rungs — these need **no API key**: only the embedder (``requires_model``). The rank check involves no
model at all.
"""

from __future__ import annotations

import pytest

from examples.common import load_config
from examples.part_iii import _conversation as conversation
from examples.part_iii.example_01 import run as without_memory
from examples.part_iii.example_02 import run as with_memory
from tarnrag import TarnRag
from tarnrag.core.engine.config import CONVERSATION_MEMORY, EmbeddingSettings
from tarnrag.core.resources.llm import Prompt

FIRST, FOLLOW_UP = conversation.TURNS


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


def test_the_two_configs_differ_only_in_the_memory_switch():
    """The diff between the two configs is the lesson — it must stay exactly one setting."""
    off = load_config(without_memory.CONFIG).model_dump()
    on = load_config(with_memory.CONFIG).model_dump()
    assert off["components"][CONVERSATION_MEMORY].pop("enabled") is False
    assert on["components"][CONVERSATION_MEMORY].pop("enabled") is True
    assert off == on  # everything else — store, embedder, pipelines, window — is identical


@pytest.mark.requires_model
async def test_example_01_without_memory_the_follow_up_misses():
    (first, follow_up), remembered = await without_memory.main()
    assert first.rank == 1 and FIRST.gold in first.result.answer  # the first question stands alone
    assert follow_up.searched == FOLLOW_UP.question  # searched exactly as typed
    assert follow_up.rank is None  # the passage holding the answer never reached the reader
    assert FOLLOW_UP.gold not in follow_up.result.answer
    assert remembered == 0  # nothing was recorded


@pytest.mark.requires_model
async def test_example_02_with_memory_the_follow_up_is_rewritten_and_hits():
    (first, follow_up), remembered = await with_memory.main()
    assert first.rank == 1 and first.searched == FIRST.question  # nothing to rewrite on the first turn
    assert follow_up.searched != FOLLOW_UP.question and "GMV" in follow_up.searched  # the frame is named
    assert follow_up.rank == 1  # …so the right spec is retrieved first
    assert FOLLOW_UP.gold in follow_up.result.answer
    assert [c.document_id for c in follow_up.result.proof[0].citations] == [FOLLOW_UP.document]
    assert remembered == 2


async def test_the_narrator_plays_each_turn_as_one_ask_and_reports_that_run():
    """Model-free (the ``hash`` embedder — no semantics, so no rank is asserted): a follow-up costs ONE
    rewrite call, what the narrator prints as searched is what that ``ask`` recorded, and the verdict is
    taken over the passages the reader was actually shown."""
    settings = load_config(with_memory.CONFIG)
    settings.embedding = EmbeddingSettings(provider="hash")
    llm = conversation.scripted_model()
    rewrites: list[Prompt] = []
    complete = llm.complete

    async def counting(prompt: Prompt):
        if "standalone_question" in (prompt.system or ""):
            rewrites.append(prompt)
        return await complete(prompt)

    llm.complete = counting
    async with TarnRag(settings, llm=llm) as tarn:
        narrator = conversation.Narrator(tarn, settings)
        await narrator.ensure_corpus()
        played = [await narrator.play(turn) for turn in conversation.TURNS]
        recorded = (await tarn.history()).value
    assert len(rewrites) == 1  # the follow-up only — and once, not again for a dry run
    assert [p.searched for p in played] == [t.standalone_question for t in recorded]
    for turn, run in zip(conversation.TURNS, played):
        if run.rank is not None:
            assert turn.gold.lower() in run.result.evidence[run.rank - 1].text.lower()


# ---------------- the scripted model is bound to what it is shown (model-free) ----------------


async def _rewrite(transcript: str, question: str) -> str:
    system = 'Reply with {"standalone_question": "..."}'
    user = f"Conversation so far:\n{transcript}\n\nLatest question: {question}"
    return (await conversation.scripted_model().complete(Prompt(system=system, user=user))).text


async def test_the_scripted_rewriter_resolves_it_only_from_the_transcript_it_is_shown():
    follow_up = "Which oil grade does it require?"
    shown = "[1] Q: Which frame is the TX-200 built on?\n    A: Cooper-Bessemer GMV frame"
    assert "the Cooper-Bessemer GMV frame" in await _rewrite(shown, follow_up)
    # No remembered answer to point at → nothing to substitute: the question comes back unchanged.
    searched_only = "[1] Q: TX-200 frame\n    (a search - no answer was recorded)"
    assert f'"{follow_up}"' in await _rewrite(searched_only, follow_up)
    # An unchecked answer (no grounding check ran) is marked, and still resolvable.
    unchecked = "[1] Q: Which frame?\n    A (unchecked - not verified against the sources): GMV frame"
    assert "the GMV frame" in await _rewrite(unchecked, follow_up)


async def test_the_scripted_reader_answers_only_from_a_passage_that_names_the_subject():
    async def read(question: str, passages: str) -> str:
        prompt = Prompt(user=f"Question: {question}\n\nPassages:\n{passages}")
        return (await conversation.scripted_model().complete(prompt)).text

    gmv = "[1] Cooper-Bessemer GMV frames require a synthetic crankcase oil of grade ISO VG 68."
    ariel = "[1] Ariel JGT compressor frames require a synthetic crankcase oil of grade ISO VG 100."
    question = "Which oil grade does the Cooper-Bessemer GMV frame require?"
    assert '"ISO VG 68"' in await read(question, gmv)
    assert "not specified" in await read(question, ariel)  # shown only another frame's spec → no answer
    assert "not specified" in await read("Which oil grade does it require?", gmv)  # names nothing
