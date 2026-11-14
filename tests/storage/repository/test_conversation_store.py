"""The ``ConversationStore`` port on the repository — the conversation layer's persisted memory.

One row per turn in ``conversation_turns`` (doc/conversation-memory-design.md §4): the store assigns
each turn its position, reads a conversation back oldest-first (optionally only the most recent N —
the short-term window), keeps conversations isolated, and keeps every turn until asked to delete.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tarnrag.contracts import ConversationStore, ConversationTurn, IndexMeta
from tarnrag.storage.repository.sqlite import SqliteRepository


def _ask(session: str, question: str, answer: str = "a", **kw) -> ConversationTurn:
    return ConversationTurn(
        session_id=session, kind="ask", question=question, standalone_question=question,
        answer=answer, grounded=True, abstained=False, **kw,
    )


async def test_the_repository_implements_the_conversation_port(repo):
    assert isinstance(repo, ConversationStore)


async def test_append_assigns_consecutive_ordinals_and_a_timestamp(repo):
    first = await repo.append_turn(_ask("s", "q0"))
    second = await repo.append_turn(_ask("s", "q1"))
    assert (first.ordinal, second.ordinal) == (0, 1)
    assert isinstance(first.created_at, datetime)
    assert first.question == "q0"  # the returned turn is the stored one, with the assigned fields filled


async def test_timestamps_are_utc_on_the_way_in_and_out(repo):
    """``created_at`` is written as UTC and read back timezone-aware, so it means the same on SQLite and
    Postgres (a server default is UTC on one and the session's local time zone on the other)."""
    before = datetime.now(UTC)
    appended = await repo.append_turn(_ask("s", "q0"))
    (stored,) = await repo.conversation_turns("s")
    assert appended.created_at.tzinfo is UTC and stored.created_at.utcoffset() == timedelta(0)
    assert stored.created_at == appended.created_at
    assert before <= stored.created_at <= datetime.now(UTC)


async def test_concurrent_appends_to_one_conversation_all_land(repo):
    """Two writers on one session (two processes resuming it) each get their own ordinal, never an error."""
    turns = await asyncio.gather(*(repo.append_turn(_ask("s", f"q{i}")) for i in range(6)))
    assert sorted(t.ordinal for t in turns) == list(range(6))
    assert len(await repo.conversation_turns("s")) == 6


async def test_an_ask_turn_round_trips_every_field(repo):
    await repo.append_turn(
        ConversationTurn(
            session_id="s", kind="ask", question="and its oil?",
            standalone_question="Which oil does the GMV frame need?",
            answer="ISO VG 68", grounded=False, abstained=True,
        )
    )
    (turn,) = await repo.conversation_turns("s")
    assert turn.kind == "ask"
    assert turn.question == "and its oil?"  # as typed
    assert turn.standalone_question == "Which oil does the GMV frame need?"  # as executed
    assert turn.answer == "ISO VG 68"
    assert turn.grounded is False and turn.abstained is True  # real bools, not 0/1
    assert turn.ordinal == 0 and isinstance(turn.created_at, datetime)


async def test_a_retrieve_turn_stores_no_answer_and_no_verdicts(repo):
    await repo.append_turn(
        ConversationTurn(session_id="s", kind="retrieve", question="pump seals", standalone_question="pump seals")
    )
    (turn,) = await repo.conversation_turns("s")
    assert turn.kind == "retrieve"
    assert turn.answer is None and turn.grounded is None and turn.abstained is None  # NULL, not False


async def test_turns_come_back_oldest_first(repo):
    for i in range(4):
        await repo.append_turn(_ask("s", f"q{i}"))
    assert [t.question for t in await repo.conversation_turns("s")] == ["q0", "q1", "q2", "q3"]


async def test_limit_keeps_the_most_recent_turns_still_oldest_first(repo):
    """The short-term window: the LAST n turns, in chronological order (not the first n)."""
    for i in range(5):
        await repo.append_turn(_ask("s", f"q{i}"))
    window = await repo.conversation_turns("s", limit=2)
    assert [t.question for t in window] == ["q3", "q4"]
    assert [t.ordinal for t in window] == [3, 4]
    assert len(await repo.conversation_turns("s", limit=99)) == 5  # a limit past the end returns all


async def test_conversations_are_isolated(repo):
    await repo.append_turn(_ask("a", "from a"))
    await repo.append_turn(_ask("b", "from b"))
    await repo.append_turn(_ask("a", "a again"))
    assert [t.question for t in await repo.conversation_turns("a")] == ["from a", "a again"]
    assert [(t.question, t.ordinal) for t in await repo.conversation_turns("b")] == [("from b", 0)]


async def test_an_unknown_conversation_is_empty(repo):
    assert await repo.conversation_turns("never-seen") == []
    assert await repo.delete_conversation("never-seen") == 0


async def test_delete_removes_only_that_conversation_and_reports_the_count(repo):
    for i in range(3):
        await repo.append_turn(_ask("a", f"q{i}"))
    await repo.append_turn(_ask("b", "keep me"))
    assert await repo.delete_conversation("a") == 3
    assert await repo.conversation_turns("a") == []
    assert len(await repo.conversation_turns("b")) == 1
    # a deleted conversation starts over from ordinal 0
    assert (await repo.append_turn(_ask("a", "fresh"))).ordinal == 0


async def test_turns_outside_the_window_are_kept(repo):
    """Retention is 'keep all, read the window': a windowed read deletes nothing."""
    for i in range(6):
        await repo.append_turn(_ask("s", f"q{i}"))
    await repo.conversation_turns("s", limit=2)
    assert len(await repo.conversation_turns("s")) == 6


async def test_limit_zero_keeps_none_and_a_negative_limit_is_refused(repo):
    """``0`` keeps nothing; a negative limit is refused here rather than read by SQLite as "no limit"
    (and refused by Postgres), so the two backends agree."""
    await repo.append_turn(_ask("s", "q0"))
    assert await repo.conversation_turns("s", limit=0) == []
    with pytest.raises(ValueError, match="limit"):
        await repo.conversation_turns("s", limit=-1)


async def test_an_unknown_kind_is_refused_by_the_schema(repo):
    with pytest.raises(IntegrityError):
        await repo.append_turn(
            ConversationTurn(session_id="s", kind="explain", question="q", standalone_question="q")
        )
    assert await repo.conversation_turns("s") == []  # the failed append left nothing behind


async def test_turns_survive_a_reconnect(tmp_path):
    """Persisted, not in-process: a second repository over the same file sees the conversation."""
    url = f"sqlite:///{tmp_path}/conv.db"
    first = SqliteRepository(url, embedding_dimension=3)
    await first.connect()
    await first.append_turn(_ask("s", "before restart"))
    await first.disconnect()

    second = SqliteRepository(url, embedding_dimension=3)
    await second.connect()
    try:
        assert [t.question for t in await second.conversation_turns("s")] == ["before restart"]
        assert (await second.append_turn(_ask("s", "after restart"))).ordinal == 1  # continues the count
    finally:
        await second.disconnect()


async def test_a_store_built_before_the_table_existed_gains_it_on_connect(tmp_path):
    """The table is additive: an existing store (here: one with the table dropped, standing in for an
    index built by an earlier version) gets it from ``connect()`` — no rebuild, and the documents and
    index identity already in the store are untouched. That is why SCHEMA_VERSION is not bumped."""
    url = f"sqlite:///{tmp_path}/old.db"
    old = SqliteRepository(url, embedding_dimension=3)
    await old.connect()
    await old.write_index_meta({"schema_version": IndexMeta.SCHEMA_VERSION})
    async with old.engine.begin() as conn:
        await conn.execute(text("DROP TABLE conversation_turns"))
    await old.disconnect()

    reopened = SqliteRepository(url, embedding_dimension=3)
    await reopened.connect()
    try:
        assert (await reopened.append_turn(_ask("s", "q"))).ordinal == 0
        assert (await reopened.index_meta())["schema_version"] == IndexMeta.SCHEMA_VERSION
    finally:
        await reopened.disconnect()
