"""ConversationContext — the runtime resources the conversation layer needs, injected at call time.

Mirrors ``RetrievalContext`` / ``GenerationContext``: ``ConversationMemory`` and its contextualizer are
pure strategies; the store (a **port**, so the package never sees the repository) and the rewriting LLM
are passed in here — which is also how tests fake them, by injection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tarnrag.contracts import ConversationStore
from tarnrag.core.resources.llm import LanguageModel


@dataclass
class ConversationContext:
    """The resources a conversation call consumes: where turns are remembered + the rewriter LLM.
    The LLM is either given (``llm``) or built on first use by ``llm_factory`` (what the facade passes),
    so only a rewrite builds it: reading or recording turns never does. There may be none (no key
    configured): a follow-up then cannot be rewritten, and the contextualizer falls back to the
    question as typed and says so."""

    store: ConversationStore
    llm: LanguageModel | None = None
    llm_factory: Callable[[], LanguageModel | None] | None = field(default=None, repr=False)

    def language_model(self) -> LanguageModel | None:
        """The rewriter LLM: ``llm``, else the one ``llm_factory`` builds (asked once), else ``None``."""
        if self.llm is None and self.llm_factory is not None:
            factory, self.llm_factory = self.llm_factory, None
            self.llm = factory()
        return self.llm
