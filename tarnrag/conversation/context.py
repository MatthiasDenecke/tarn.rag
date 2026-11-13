"""ConversationContext — the runtime resources the conversation layer needs, injected at call time.

Mirrors ``RetrievalContext`` / ``GenerationContext``: ``ConversationMemory`` and its contextualizer are
pure strategies; the store (a **port**, so the package never sees the repository) and the rewriting LLM
are passed in here — which is also how tests fake them, by injection.
"""

from __future__ import annotations

from dataclasses import dataclass

from tarnrag.contracts import ConversationStore
from tarnrag.core.resources.llm import LanguageModel


@dataclass
class ConversationContext:
    """The resources a conversation call consumes: where turns are remembered + the rewriter LLM.
    ``llm`` may be ``None`` (no key configured): a follow-up then cannot be rewritten, and the
    contextualizer falls back to the question as typed and says so."""

    store: ConversationStore
    llm: LanguageModel | None = None
