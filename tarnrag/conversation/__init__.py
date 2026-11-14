"""Conversation domain — the short-term conversational context layer.

A top-level package next to ``ingestion/``, ``retrieval/`` and ``generation/``. It makes a follow-up
question ("which oil grade does *it* require?") stand on its own by rewriting it against the last few
turns of the conversation, *before* retrieval or generation run — so every existing retriever, reasoner
and grounding check works unchanged, on a question that carries its own context.

Dependency rule: ``conversation → core + contracts``, nothing else. It imports neither ``retrieval`` nor
``generation`` and neither imports it; the ``TarnRag`` facade (the composition root) is the only code
that knows all of them, and it applies this layer in front of ``retrieve`` / ``explain`` / ``ask``.
Turns are persisted through the ``ConversationStore`` port (implemented by the repository — one store).

``ConversationMemory`` is the container Component built from ``Settings.components[CONVERSATION_MEMORY]``
(off by default); ``QueryContextualizer`` is its strategy seam. See ``doc/conversation-memory-design.md``.
"""

from tarnrag.contracts import ConversationStore, ConversationTurn
from tarnrag.conversation.components.contextualizer import (
    CondenseContextualizer,
    QueryContextualizer,
)
from tarnrag.conversation.context import ConversationContext
from tarnrag.conversation.memory import ConversationMemory
from tarnrag.conversation.types import CONTEXTUALIZED, Contextualization

__all__ = [
    "CONTEXTUALIZED",
    "CondenseContextualizer",
    "Contextualization",
    "ConversationContext",
    "ConversationMemory",
    "ConversationStore",
    "ConversationTurn",
    "QueryContextualizer",
]
