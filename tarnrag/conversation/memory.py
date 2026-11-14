"""ConversationMemory — the short-term conversational context layer, as a container Component.

Built from the ``CONVERSATION_MEMORY`` spec in ``Settings.components``. It owns the **window policy**
(the last ``max_turns`` turns, each answer cut to ``max_answer_chars``) and delegates *how* that window
is used to its ``QueryContextualizer`` child. The flow it supports, driven by the ``TarnRag`` facade:

    contextualize (read the window → standalone question)  →  run the operation  →  remember (append)

It is a pure strategy: it holds no turns and no session. The session id is an argument of every call
and the turns live behind the ``ConversationStore`` port, so one component serves any conversation.
``enabled`` is honoured *here*, not only by the facade — a disabled memory reads nothing, calls no LLM
and writes nothing, whoever drives it. See ``doc/conversation-memory-design.md``.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Literal

from bausatz import Component, ComponentFactory
from pydantic import Field

from tarnrag.contracts import ConversationTurn
from tarnrag.conversation.components.contextualizer import (
    CondenseContextualizer,
    QueryContextualizer,
)
from tarnrag.conversation.context import ConversationContext
from tarnrag.conversation.types import Contextualization

_ELLIPSIS = "…"


class ConversationMemory(Component):
    """Remember the recent turns of a conversation and use them to make a follow-up stand on its own."""

    class Config(Component.Config):
        class_name: Literal["conversation_memory"] = "conversation_memory"
        # The on/off switch. Off (the default) is exactly the stateless behaviour: no read, no LLM
        # call, no write.
        enabled: bool = False
        # Which conversation a facade joins on open(): "" → a fresh one each time; an id → resume it.
        session_id: str = ""
        max_turns: int = Field(default=6, ge=1)  # the short-term window: how many recent turns are read
        # Each remembered answer is cut to this many characters before the contextualizer sees it.
        max_answer_chars: int = Field(default=600, ge=1)
        contextualizer: QueryContextualizer.Config = Field(default_factory=CondenseContextualizer.Config)

    config: ConversationMemory.Config

    def __init__(self, config: ConversationMemory.Config) -> None:
        super().__init__(config)
        self._contextualizer: QueryContextualizer | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        self._contextualizer = factory.build_as(self.config.contextualizer, QueryContextualizer)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def new_session_id() -> str:
        """A fresh conversation id (what a facade uses when none is configured or passed)."""
        return uuid.uuid4().hex

    async def contextualize(
        self, question: str, session_id: str, ctx: ConversationContext
    ) -> Contextualization:
        """The question as it should be executed, given the conversation so far. Read-only. With memory
        off, or nothing remembered yet, the question is returned as typed **without an LLM call** — the
        first turn of a conversation costs nothing extra."""
        window = await self.window(session_id, ctx)
        if not window:
            return Contextualization(original=question, standalone=question)
        self.ensure_built()
        return await self._contextualizer.contextualize(question, window, ctx)

    async def window(self, session_id: str, ctx: ConversationContext) -> list[ConversationTurn]:
        """The short-term window a contextualizer is shown: the last ``max_turns`` turns, oldest first,
        each answer cut to ``max_answer_chars``. Empty when memory is off."""
        if not self.enabled:
            return []
        turns = await ctx.store.conversation_turns(session_id, limit=self.config.max_turns)
        return [self._clip(turn) for turn in turns]

    async def remember(self, turn: ConversationTurn, ctx: ConversationContext) -> ConversationTurn:
        """Append ``turn`` to its conversation; return it as stored (ordinal + timestamp assigned).
        With memory off nothing is written and the turn comes back unchanged."""
        if not self.enabled:
            return turn
        return await ctx.store.append_turn(turn)

    async def history(
        self, session_id: str, ctx: ConversationContext, last: int | None = None
    ) -> list[ConversationTurn]:
        """The stored conversation, oldest first and **unclipped** — the transcript, not the window
        (turns outside the window are kept; they are simply never shown to the contextualizer).
        ``last`` keeps the most recent ``last`` turns. Empty when memory is off."""
        if not self.enabled:
            return []
        return await ctx.store.conversation_turns(session_id, limit=last)

    async def forget(self, session_id: str, ctx: ConversationContext) -> int:
        """Delete one conversation's turns from the store; return how many were removed."""
        if not self.enabled:
            return 0
        return await ctx.store.delete_conversation(session_id)

    def _clip(self, turn: ConversationTurn) -> ConversationTurn:
        limit = self.config.max_answer_chars
        if turn.answer is None or len(turn.answer) <= limit:
            return turn
        return replace(turn, answer=turn.answer[:limit].rstrip() + _ELLIPSIS)
