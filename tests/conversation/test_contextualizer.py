"""The ``condense`` contextualizer — one LLM call that makes a follow-up stand on its own."""

import json

import pytest
from bausatz import ComponentFactory

from tarnrag.contracts import ConversationTurn
from tarnrag.conversation import CondenseContextualizer, ConversationContext
from tarnrag.conversation.components.contextualizer import NO_LLM, UNUSABLE_REPLY


def _condense(**config) -> CondenseContextualizer:
    return ComponentFactory.get().create({"class_name": "condense", **config})


def _reply(standalone: str) -> str:
    return json.dumps({"standalone_question": standalone})


async def test_a_follow_up_is_rewritten_into_the_models_standalone_question(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("Which oil grade does the GMV frame require?"))
    history = [ask_turn("Which frame is the TX-200 built on?", "Cooper-Bessemer GMV")]
    result = await _condense().contextualize(
        "Which oil grade does it require?", history, ConversationContext(store, llm)
    )
    assert result.original == "Which oil grade does it require?"  # as typed — kept, never overwritten
    assert result.standalone == "Which oil grade does the GMV frame require?"  # as executed
    assert result.rewritten is True
    assert result.turns_used == 1 and result.producer == "condense" and result.fallback is None
    assert len(llm.prompts) == 1  # exactly one LLM call per follow-up


async def test_the_prompt_lays_out_the_window_then_the_latest_question(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    history = [
        ask_turn("Which frame is the TX-200 built on?", "Cooper-Bessemer GMV"),
        ask_turn("and its oil?", "ISO VG 68", standalone="Which oil grade does the GMV frame require?"),
    ]
    await _condense().contextualize("and for the Ariel JGT?", history, ConversationContext(store, llm))
    user = llm.prompts[0].user
    assert user.index("[1] Q: Which frame is the TX-200 built on?") < user.index("[2] Q:")  # oldest first
    assert "A: Cooper-Bessemer GMV" in user and "A: ISO VG 68" in user
    assert user.rstrip().endswith("Latest question: and for the Ariel JGT?")
    # An earlier turn appears as its STANDALONE question — already resolved, so a chain of references
    # stays resolvable — never as the raw follow-up that was typed.
    assert "Which oil grade does the GMV frame require?" in user
    assert "and its oil?" not in user


async def test_the_instruction_forbids_answering_and_adding_facts(store, recording_llm, ask_turn):
    """The rewrite must stay a rewrite: a rewriter that answered, or decorated the question with facts,
    would carry un-cited content past the grounding check."""
    llm = recording_llm(_reply("x"))
    await _condense().contextualize("q", [ask_turn("p")], ConversationContext(store, llm))
    system = llm.prompts[0].system
    assert "Do NOT answer" in system and "Do NOT add facts" in system
    assert "standalone_question" in system  # the strict-JSON reply contract


async def test_the_rewrite_call_is_capped_by_max_tokens(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    await _condense(max_tokens=64).contextualize("q", [ask_turn("p")], ConversationContext(store, llm))
    assert llm.prompts[0].max_tokens == 64


async def test_the_rewrite_runs_at_temperature_zero_unless_configured(store, recording_llm, ask_turn):
    """Pinned, so a rewrite is repeatable: explain's dry run then shows what the next ask searches."""
    llm = recording_llm(_reply("x"))
    await _condense().contextualize("q", [ask_turn("p")], ConversationContext(store, llm))
    await _condense(temperature=None).contextualize("q", [ask_turn("p")], ConversationContext(store, llm))
    assert [p.temperature for p in llm.prompts] == [0.0, None]  # None → the configured LLM default


async def test_unverified_and_refused_answers_are_marked_for_the_rewriter(store, recording_llm, ask_turn):
    llm = recording_llm(_reply("x"))
    history = [
        ask_turn("q1", "a solid answer"),
        ask_turn("q2", "a shaky answer", grounded=False),
        ask_turn("q3", "I can't answer that.", grounded=False, abstained=True),
        ask_turn("q4", "an unchecked answer", grounded=None),  # no grounding check ran
    ]
    await _condense().contextualize("next", history, ConversationContext(store, llm))
    user = llm.prompts[0].user
    assert "A: a solid answer" in user  # a verified answer carries no marker
    assert "A (unverified" in user and "a shaky answer" in user
    assert "A (the assistant declined to answer): I can't answer that." in user
    assert "A (unchecked - not verified against the sources): an unchecked answer" in user


async def test_a_retrieve_turn_is_shown_as_a_search_without_an_answer(store, recording_llm):
    llm = recording_llm(_reply("x"))
    searched = ConversationTurn(
        session_id="s", kind="retrieve", question="pump seals", standalone_question="pump seals"
    )
    await _condense().contextualize("how often are they replaced?", [searched], ConversationContext(store, llm))
    user = llm.prompts[0].user
    assert "[1] Q: pump seals" in user and "no answer was recorded" in user
    assert "A:" not in user


async def test_an_already_standalone_question_comes_back_unchanged(store, recording_llm, ask_turn):
    question = "What does API 653 cover?"
    llm = recording_llm(_reply(question))
    result = await _condense().contextualize(question, [ask_turn("p")], ConversationContext(store, llm))
    assert result.standalone == question and result.rewritten is False
    assert result.fallback is None  # the model ran and chose to keep it — not a failure
    assert result.annotations() == []  # nothing was rewritten, so the query is exactly the stateless one


async def test_a_reply_wrapped_in_prose_still_parses(store, recording_llm, ask_turn):
    llm = recording_llm('Sure! {"standalone_question": "  What is X?  "} Hope that helps.')
    result = await _condense().contextualize("and X?", [ask_turn("p")], ConversationContext(store, llm))
    assert result.standalone == "What is X?"  # extracted and stripped


@pytest.mark.parametrize(
    "reply",
    [
        "I think they mean the GMV frame.",  # not JSON at all
        '{"question": "wrong key"}',  # JSON, but not the contract
        '{"standalone_question": ""}',  # empty
        '{"standalone_question": "   "}',  # blank
        '{"standalone_question": 42}',  # not a string
        '["a", "list"]',  # JSON, but not an object
        "",  # nothing
    ],
)
async def test_an_unusable_reply_falls_back_to_the_question_as_typed(store, recording_llm, ask_turn, reply):
    llm = recording_llm(reply)
    result = await _condense().contextualize("and its oil?", [ask_turn("p")], ConversationContext(store, llm))
    assert result.standalone == "and its oil?" and result.rewritten is False
    assert result.fallback == UNUSABLE_REPLY  # the reason is recorded — never a silent pass-through
    assert result.annotations() == []


async def test_without_an_llm_it_falls_back_and_says_why(store, ask_turn):
    result = await _condense().contextualize("and its oil?", [ask_turn("p")], ConversationContext(store, None))
    assert result.standalone == "and its oil?"
    assert result.fallback == NO_LLM and result.turns_used == 1


async def test_a_failing_llm_call_propagates(store, ask_turn):
    """A transport/auth error is a misconfiguration: it fails loudly rather than degrading."""

    class Broken:
        async def complete(self, prompt):
            raise ConnectionError("endpoint unreachable")

    with pytest.raises(ConnectionError):
        await _condense().contextualize("q", [ask_turn("p")], ConversationContext(store, Broken()))


async def test_a_rewrite_is_annotated_as_generated(store, recording_llm, ask_turn):
    """The query built from a rewrite carries the original wording, flagged non-deterministic — the
    channel the design uses for LLM findings on a query (never silently trusted)."""
    llm = recording_llm(_reply("Which oil grade does the GMV frame require?"))
    result = await _condense().contextualize(
        "and its oil?", [ask_turn("p"), ask_turn("q")], ConversationContext(store, llm)
    )
    (annotation,) = result.annotations()
    assert annotation.producer == "condense" and annotation.type == "contextualized"
    assert annotation.value == {"original": "and its oil?", "turns_used": 2}
    assert annotation.deterministic is False
