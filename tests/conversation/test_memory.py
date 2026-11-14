"""``ConversationMemory`` — the window policy + the on/off switch, over the ``ConversationStore`` port."""

import json

import pytest
from bausatz import ComponentFactory
from pydantic import ValidationError

from tarnrag.conversation import (
    CondenseContextualizer,
    Contextualization,
    ConversationContext,
    ConversationMemory,
)


def _memory(**config) -> ConversationMemory:
    spec = {"class_name": "conversation_memory", "enabled": True, **config}
    return ComponentFactory.get().create_as(spec, ConversationMemory)


def _reply(standalone: str) -> str:
    return json.dumps({"standalone_question": standalone})


# ---------------- the first turn / the switch ----------------


async def test_the_llm_is_built_only_when_a_follow_up_needs_a_rewrite(store, recording_llm, ask_turn):
    """Reading and recording turns, and a first turn with nothing to resolve, never build the LLM: a
    misconfigured one can then only fail the rewrite that needs it."""
    built = []

    def build():
        built.append("llm")
        return recording_llm(_reply("x"))

    memory = _memory()
    ctx = ConversationContext(store, llm_factory=build)
    await memory.contextualize("first", "s", ctx)
    await memory.remember(ask_turn("first"), ctx)
    await memory.history("s", ctx)
    assert built == []
    await memory.contextualize("and then?", "s", ctx)
    await memory.contextualize("and now?", "s", ctx)
    assert built == ["llm"]  # built once, by the first rewrite


async def test_the_first_turn_is_used_as_typed_without_an_llm_call(store, recording_llm):
    llm = recording_llm(_reply("should never be used"))
    result = await _memory().contextualize("Which frame is the TX-200 built on?", "s", ConversationContext(store, llm))
    assert result == Contextualization(
        original="Which frame is the TX-200 built on?", standalone="Which frame is the TX-200 built on?"
    )
    assert llm.prompts == []  # nothing remembered yet → nothing to resolve → no cost
    assert result.annotations() == []


async def test_a_follow_up_is_resolved_against_what_was_remembered(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("Which oil grade does the GMV frame require?"))
    memory, ctx = _memory(), ConversationContext(store, llm)
    await memory.remember(ask_turn("Which frame is the TX-200 built on?", "Cooper-Bessemer GMV"), ctx)
    result = await memory.contextualize("Which oil grade does it require?", "s", ctx)
    assert result.standalone == "Which oil grade does the GMV frame require?"
    assert "Cooper-Bessemer GMV" in llm.prompts[0].user  # the remembered answer reached the rewriter


async def test_disabled_memory_reads_nothing_calls_nothing_and_writes_nothing(store, recording_llm, ask_turn):
    """Off is the stateless behaviour exactly — honoured by the component itself, whoever drives it."""
    llm = recording_llm(_reply("should never be used"))
    memory, ctx = _memory(enabled=False), ConversationContext(store, llm)
    store.turns["s"] = [ask_turn("an earlier turn", "that must not be consulted")]

    result = await memory.contextualize("and its oil?", "s", ctx)
    assert result.standalone == "and its oil?" and result.turns_used == 0
    turn = ask_turn("and its oil?")
    assert await memory.remember(turn, ctx) is turn  # handed back unstored
    assert await memory.history("s", ctx) == []
    assert await memory.forget("s", ctx) == 0

    assert store.reads == 0 and llm.prompts == []
    assert len(store.turns["s"]) == 1  # nothing appended, nothing deleted


def test_memory_is_off_by_default():
    assert ComponentFactory.get().create({"class_name": "conversation_memory"}).enabled is False


# ---------------- the window ----------------


async def test_the_window_is_the_last_n_turns_oldest_first(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    memory, ctx = _memory(max_turns=2), ConversationContext(store, llm)
    for i in range(5):
        await memory.remember(ask_turn(f"question {i}", f"answer {i}"), ctx)
    result = await memory.contextualize("and then?", "s", ctx)
    user = llm.prompts[0].user
    assert result.turns_used == 2
    assert "question 2" not in user  # outside the window: forgotten by the rewriter…
    assert user.index("[1] Q: question 3") < user.index("[2] Q: question 4")  # …the last two, in order
    assert len(await memory.history("s", ctx)) == 5  # …but kept in the store (keep all, read the window)


async def test_a_long_answer_is_clipped_in_the_window_but_not_in_the_transcript(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    memory, ctx = _memory(max_answer_chars=10), ConversationContext(store, llm)
    await memory.remember(ask_turn("q", "0123456789ABCDEFGHIJ"), ctx)
    await memory.contextualize("next", "s", ctx)
    assert "A: 0123456789…" in llm.prompts[0].user and "ABCDEFGHIJ" not in llm.prompts[0].user
    (stored,) = await memory.history("s", ctx)
    assert stored.answer == "0123456789ABCDEFGHIJ"  # the stored turn is never altered


async def test_an_answer_within_the_cap_is_shown_whole(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    memory, ctx = _memory(max_answer_chars=10), ConversationContext(store, llm)
    await memory.remember(ask_turn("q", "0123456789"), ctx)
    await memory.contextualize("next", "s", ctx)
    assert "A: 0123456789\n" in llm.prompts[0].user  # exactly at the cap: no ellipsis


# ---------------- sessions / history / forget ----------------


async def test_conversations_do_not_see_each_other(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    memory, ctx = _memory(), ConversationContext(store, llm)
    await memory.remember(ask_turn("about pumps", session="a"), ctx)
    result = await memory.contextualize("and its seals?", "b", ctx)
    assert result.turns_used == 0 and llm.prompts == []  # conversation b has no history of its own


async def test_history_returns_the_transcript_and_last_trims_it(store, ask_turn):
    memory, ctx = _memory(max_turns=1), ConversationContext(store)
    for i in range(3):
        await memory.remember(ask_turn(f"q{i}"), ctx)
    assert [t.question for t in await memory.history("s", ctx)] == ["q0", "q1", "q2"]  # not the window
    assert [t.question for t in await memory.history("s", ctx, last=1)] == ["q2"]


async def test_forget_deletes_the_conversation_and_the_next_turn_starts_clean(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    memory, ctx = _memory(), ConversationContext(store, llm)
    await memory.remember(ask_turn("q0"), ctx)
    await memory.remember(ask_turn("q1"), ctx)
    assert await memory.forget("s", ctx) == 2
    result = await memory.contextualize("and its oil?", "s", ctx)
    assert result.turns_used == 0 and llm.prompts == []


def test_new_session_ids_are_unique():
    assert ConversationMemory.new_session_id() != ConversationMemory.new_session_id()


# ---------------- the spec ----------------


def test_the_spec_validates_as_a_whole_tree_with_its_contextualizer_child():
    config = ComponentFactory.get().validate(
        {"class_name": "conversation_memory", "contextualizer": {"class_name": "condense", "max_tokens": 99}}
    )
    assert isinstance(config.contextualizer, CondenseContextualizer.Config)
    assert config.contextualizer.max_tokens == 99
    assert (config.max_turns, config.max_answer_chars, config.session_id) == (6, 600, "")


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        ({"max_turns": 0}, "max_turns"),
        ({"max_answer_chars": 0}, "max_answer_chars"),
        ({"window": 3}, "window"),  # an unknown option is an error, not ignored
        ({"contextualizer": {"class_name": "rrf"}}, "contextualizer"),  # a fuser is not a contextualizer
        ({"contextualizer": {"class_name": "condense", "max_tokens": 0}}, "max_tokens"),
    ],
)
def test_a_bad_spec_fails_at_validation_naming_the_field(spec, path):
    with pytest.raises(ValidationError, match=path):
        ComponentFactory.get().validate({"class_name": "conversation_memory", **spec})


def test_the_spec_round_trips_through_to_dict():
    factory = ComponentFactory.get()
    memory = _memory(session_id="shift-a", max_turns=3)
    assert factory.create(memory.to_dict()).to_dict() == memory.to_dict()
    assert memory.to_dict()["contextualizer"]["class_name"] == "condense"
