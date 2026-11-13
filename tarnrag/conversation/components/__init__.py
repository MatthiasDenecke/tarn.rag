"""Conversation components — the seams ``ConversationMemory`` composes.

The ``QueryContextualizer`` (how the remembered turns are *used*: ``CondenseContextualizer`` rewrites a
follow-up into a standalone question with one LLM call). Each self-registers with the global
``ComponentFactory`` on import. Import from the module
(``from tarnrag.conversation.components.contextualizer import CondenseContextualizer``).
"""
