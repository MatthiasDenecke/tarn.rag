"""Shared fakes for the conversation tests: an in-memory ``ConversationStore`` + a recording LLM.

The store double implements the *port* (not the repository), which is the point: the conversation
package must work against any ``ConversationStore``. The repository's own implementation is covered in
``tests/storage/repository/test_conversation_store.py``.
"""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tarnrag.contracts import ConversationStore, ConversationTurn
from tarnrag.core.resources.llm import Completion, LanguageModel, Prompt


class InMemoryConversationStore(ConversationStore):
    """A ``ConversationStore`` double holding turns in a dict; counts reads so a test can assert that
    a disabled memory never touches the store."""

    def __init__(self) -> None:
        self.turns: dict[str, list[ConversationTurn]] = {}
        self.reads = 0

    async def append_turn(self, turn: ConversationTurn) -> ConversationTurn:
        bucket = self.turns.setdefault(turn.session_id, [])
        stored = replace(turn, ordinal=len(bucket), created_at=datetime.now(UTC))
        bucket.append(stored)
        return stored

    async def conversation_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        self.reads += 1
        turns = self.turns.get(session_id, [])
        return list(turns if limit is None else turns[-limit:])

    async def delete_conversation(self, session_id: str) -> int:
        return len(self.turns.pop(session_id, []))


class RecordingLLM(LanguageModel):
    """A ``LanguageModel`` double: replies with a fixed text and records every prompt it was sent."""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.prompts: list[Prompt] = []

    def identity(self) -> str:
        return "recording"

    async def complete(self, prompt: Prompt) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.reply)


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


@pytest.fixture
def recording_llm():
    """Factory: ``recording_llm('{"standalone_question": "…"}')``."""
    return RecordingLLM


@pytest.fixture
def ask_turn():
    """Factory for an ``ask`` turn (grounded, not abstained, unless overridden)."""

    def _make(question, answer="an answer", *, session="s", standalone=None, **overrides):
        fields = {"grounded": True, "abstained": False, **overrides}
        return ConversationTurn(
            session_id=session, kind="ask", question=question,
            standalone_question=standalone or question, answer=answer, **fields,
        )

    return _make
