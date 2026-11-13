"""The QueryContextualizer seam — turn a follow-up into a question that stands on its own.

A ``QueryContextualizer`` is a config-driven ``Component`` (a pure strategy; the LLM is injected via the
``ConversationContext``). Given the latest question and the remembered turns, it returns a
``Contextualization``: the question as it should be *executed*. It works on plain text — it never sees a
``Query``, a passage, or an engine — so the conversation package depends only on ``core`` + ``contracts``.

``CondenseContextualizer`` (``condense``) is the shipped strategy: one LLM call that resolves what the
latest question refers to (a pronoun, an omitted subject, "and for X?") against the conversation and
returns the standalone question. It is deliberately *only* a rewrite — the conversation never reaches
the reader, so an answer is still built from, and verified against, retrieved passages alone.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from bausatz import Component
from pydantic import Field

from tarnrag.contracts import ConversationTurn
from tarnrag.conversation.context import ConversationContext
from tarnrag.conversation.types import Contextualization
from tarnrag.core.parsing import extract_json
from tarnrag.core.resources.llm import Prompt

# The rewrite prompt. Scoped tightly on purpose: resolve references, nothing else — a rewriter that
# answers, or decorates the question with facts, would smuggle un-cited content past the grounding check.
_CONDENSE_SYSTEM = (
    "You rewrite the LATEST question of a conversation so that it can be understood WITHOUT the "
    "conversation.\n"
    "Use the earlier turns only to resolve what the latest question refers to: pronouns (it, they, "
    'that one), an omitted subject, an elliptical follow-up ("and for X?").\n'
    "Rules:\n"
    "- Do NOT answer the question.\n"
    "- Do NOT add facts, constraints or details that are not in the conversation.\n"
    "- Keep identifiers, part numbers and names exactly as written.\n"
    "- If the latest question is already self-contained, return it unchanged.\n"
    "- An answer marked as unverified or declined is less reliable than the others.\n\n"
    'Reply with a SINGLE JSON object and nothing else: {"standalone_question": "<the rewritten question>"}'
)

# Why a follow-up was used as typed — surfaced to the caller (the facade reports it as a WARNING).
NO_LLM = "no language model is available to resolve it"
UNUSABLE_REPLY = "the rewrite reply was not a usable standalone question"


class QueryContextualizer(Component):
    """Port: produce the standalone form of ``question`` given the remembered ``history``."""

    class Config(Component.Config):
        """Base contextualizer config; concrete contextualizers pin ``class_name``."""

    @abstractmethod
    async def contextualize(
        self, question: str, history: list[ConversationTurn], ctx: ConversationContext
    ) -> Contextualization:
        """The question as it should be executed. ``history`` is the short-term window, oldest first,
        and non-empty (``ConversationMemory`` never calls a contextualizer with nothing to resolve
        against). Never raises for an unusable model reply — it falls back to the question as typed and
        records why in ``Contextualization.fallback``; a failing LLM *call* propagates."""


class CondenseContextualizer(QueryContextualizer):
    """Rewrite a follow-up into a standalone question with one LLM call (see the module docstring).
    Degrades to the question as typed — with the reason recorded — when there is no LLM or its reply is
    not the expected JSON; a transport/auth error from the call itself propagates (a misconfiguration
    fails loudly, as the ``multi_query`` / ``hyde`` retrievers do)."""

    class Config(QueryContextualizer.Config):
        class_name: Literal["condense"] = "condense"
        max_tokens: int = Field(default=256, ge=1)  # cap on the rewrite call — a rewrite is one sentence

    config: CondenseContextualizer.Config

    async def contextualize(
        self, question: str, history: list[ConversationTurn], ctx: ConversationContext
    ) -> Contextualization:
        producer = self.config.class_name
        if ctx.llm is None:
            return Contextualization(question, question, len(history), producer, fallback=NO_LLM)
        completion = await ctx.llm.complete(
            Prompt(
                system=_CONDENSE_SYSTEM,
                user=self.render(question, history),
                max_tokens=self.config.max_tokens,
            )
        )
        data = extract_json(completion.text)
        raw = data.get("standalone_question") if isinstance(data, dict) else None
        standalone = raw.strip() if isinstance(raw, str) else ""
        if not standalone:
            return Contextualization(question, question, len(history), producer, fallback=UNUSABLE_REPLY)
        return Contextualization(question, standalone, len(history), producer)

    @staticmethod
    def render(question: str, history: list[ConversationTurn]) -> str:
        """The user prompt: the window as numbered turns, then the latest question. Each earlier turn
        shows its **standalone** question (already resolved, so a chain of references — "and for that
        one?" after "and its oil?" — stays resolvable), then its answer; an unverified or refused answer
        is marked, and a ``retrieve`` turn is shown as a search that recorded no answer."""
        lines = ["Conversation so far:"]
        for number, turn in enumerate(history, 1):
            lines.append(f"[{number}] Q: {turn.standalone_question}")
            lines.append(f"    {CondenseContextualizer._answer_line(turn)}")
        lines += ["", f"Latest question: {question}"]
        return "\n".join(lines)

    @staticmethod
    def _answer_line(turn: ConversationTurn) -> str:
        if turn.answer is None:
            return "(a search - no answer was recorded)"
        if turn.abstained:
            return f"A (the assistant declined to answer): {turn.answer}"
        if turn.grounded is False:
            return f"A (unverified - not fully supported by the sources): {turn.answer}"
        return f"A: {turn.answer}"
