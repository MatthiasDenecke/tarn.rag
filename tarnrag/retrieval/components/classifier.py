"""The QueryClassifier seam — populate a query's classification (``query_type`` + ``annotations``).

A ``QueryClassifier`` is a config-driven ``Component`` that writes its findings onto the ``Query`` the
same way an ``Enricher`` annotates a chunk: it sets the cheap ``query_type`` route key and appends rich
``Annotation``s (each carrying the ``deterministic`` flag, so an LLM classifier's findings are flagged,
never silently trusted). Classification is opt-in:

- ``GenericQueryClassifier`` — the **default**; tags every query with one constant ``query_type``
  (``"generic"``) and records the annotation, so a router with no real classifier still **guarantees a
  query_type annotation exists** while routing everything to its ``default`` route.
- ``StructuralQueryClassifier`` — the first real, **domain-independent** classifier: a heuristic over the
  query's *form* (not its subject), → ``query_type ∈ {lexical, semantic}``.

The ``RoutingRetrievalPipeline`` runs the configured classifier before dispatching on ``query.query_type``;
downstream consumers (the eval harness, logging, later the ``Reasoner``) read the same fields off the
contract. Domain-specific classifiers are added later as more components on this seam — no base edits.
"""

from __future__ import annotations

import string
from abc import abstractmethod
from typing import Any, Literal

from bausatz import Component

from tarnrag.contracts import Annotation, Geometry
from tarnrag.core.text import looks_like_identifier
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class QueryClassifier(Component):
    """Port: classify a ``Query`` in place — set ``query.query_type`` and/or append ``query.annotations``.
    Subclasses call ``_annotate`` rather than touching ``.annotations`` directly, so the ``producer`` is
    recorded automatically (the protected helper analog of ``Enricher.annotate``)."""

    class Config(Component.Config):
        """Base classifier config; concrete classifiers pin ``class_name``."""

    @abstractmethod
    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        """Annotate ``query`` in place: set its ``query_type`` and/or append annotations. ``ctx`` is
        available for classifiers that need the embedder/store/LLM (the structural one ignores it)."""

    def _annotate(
        self,
        query: Query,
        type: str,
        value: dict[str, Any],
        *,
        span: Geometry | None = None,
        deterministic: bool = True,
    ) -> None:
        """Append one finding by this classifier to ``query.annotations``; ``producer`` is filled in.
        ``span`` may mark the sub-region of the query text it applies to (char offsets into ``query.text``)."""
        query.annotations.append(
            Annotation(
                producer=self.config.class_name,
                type=type,
                value=value,
                span=span,
                deterministic=deterministic,
            )
        )


class GenericQueryClassifier(QueryClassifier):
    """The default classifier: tag every query with one constant ``query_type`` (``"generic"`` by default)
    and record the matching annotation — so a router with no real classifier configured still **guarantees
    a query_type annotation exists** (uniform with the structural one), while routing everything to its
    ``default`` route (no ``"generic"`` route ⇒ the fallthrough). A caller-supplied ``query_type`` still
    wins, since the router only invokes a classifier when ``query_type`` is empty."""

    class Config(QueryClassifier.Config):
        class_name: Literal["generic"] = "generic"
        query_type: str = "generic"  # the constant label assigned to every query

    config: GenericQueryClassifier.Config

    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        query.query_type = self.config.query_type
        self._annotate(query, "query_classification", {"label": self.config.query_type})


# Compact, domain-independent word lists (English defaults; override via Config). They lean English, but
# the exact-match cues (identifiers / quotes / length) are language-independent and carry weight regardless.
# The identifier cue itself lives in ``core.text`` — shared with the sparse-query builders, so what this
# classifier labels lexical is exactly what the FTS builder phrase-requires.
_QUESTION_WORDS = frozenset(
    [
        "who", "what", "when", "where", "why", "how", "which", "whose", "whom", "is", "are", "was",
        "were", "do", "does", "did", "can", "could", "should", "would", "will", "may", "might",
        "has", "have", "had"
    ]
)
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but", "with", "at", "by",
        "from", "as", "into", "about", "than", "that", "this", "these", "those", "be", "been",
        "being", "it", "its", "i", "we", "you", "they", "he", "she", "my", "your", "our", "their",
        "over", "under", "up", "down", "out"
    ]
)


class StructuralQueryClassifier(QueryClassifier):
    """Domain-independent heuristic: classify a query by its *form* (keyword/exact vs natural-language)
    rather than its subject. Signals — interrogative form, function-word ratio, exact-match cues
    (double-quoted spans / identifiers / acronyms) — map to ``query_type ∈ {lexical, semantic}``, and the
    features found are recorded as an annotation. Deterministic, LLM-free, C++-portable;
    labels / threshold / word-lists are config. The decision is a short ordered rule, each clause
    defensible on its own:

    1. a quoted span or an identifier ⇒ ``lexical`` (explicit exact-match intent);
    2. otherwise an interrogative or function-word-heavy phrasing ⇒ ``semantic`` (natural language);
    3. otherwise (content words, no function words) ⇒ ``lexical`` (a keyword query).
    """

    class Config(QueryClassifier.Config):
        class_name: Literal["structural"] = "structural"
        lexical_type: str = "lexical"
        semantic_type: str = "semantic"
        function_word_threshold: float = 0.25  # fn-word ratio at/above which a phrase reads as semantic
        stopwords: list[str] | None = None  # None ⇒ built-in English set
        question_words: list[str] | None = None  # None ⇒ built-in English set

    config: StructuralQueryClassifier.Config

    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        cfg = self.config
        stop = frozenset(cfg.stopwords) if cfg.stopwords is not None else _STOPWORDS
        qwords = frozenset(cfg.question_words) if cfg.question_words is not None else _QUESTION_WORDS

        text = query.text.strip()
        raw_tokens = text.split()
        words = [w for w in (t.strip(string.punctuation).lower() for t in raw_tokens) if w]
        n = len(words)

        fn_ratio = sum(w in stop for w in words) / n if n else 0.0
        interrogative = bool(words) and (words[0] in qwords or text.endswith("?"))
        has_quotes = text.count('"') >= 2
        identifiers = [t for t in raw_tokens if looks_like_identifier(t)]

        if has_quotes or identifiers:
            label = cfg.lexical_type
        elif interrogative or fn_ratio >= cfg.function_word_threshold:
            label = cfg.semantic_type
        else:
            label = cfg.lexical_type

        query.query_type = label
        self._annotate(
            query,
            "query_classification",
            {
                "label": label,
                "interrogative": interrogative,
                "function_word_ratio": round(fn_ratio, 3),
                "n_tokens": n,
                "has_quotes": has_quotes,
                "identifiers": identifiers,
            },
        )


# The intent taxonomy (PP-1: the production-pipeline classes) with compact, overridable cue lists.
# Ordered by routing consequence: aggregation first (the structured-data gate — numeric questions
# should not ride the narrative path), then comparison / troubleshooting / policy / summarization;
# no cue ⇒ lookup. Multi-word cues match as substrings; single words on word boundaries.
_INTENT_CUES: dict[str, tuple[str, ...]] = {
    "aggregation": (
        "how many", "how much", "count", "total", "sum", "average", "mean", "percentage",
        "percent", "ratio", "rank", "largest", "smallest", "highest", "lowest",
    ),
    "comparison": (
        "compare", "comparison", "versus", "vs", "difference between", "differ", "better than",
        "worse than", "more than", "less than",
    ),
    "troubleshooting": (
        "error", "errors", "fail", "fails", "failed", "failure", "not working", "does not work",
        "fix", "troubleshoot", "issue", "warning", "crash", "crashes",
    ),
    "policy": (
        "allowed", "permitted", "policy", "policies", "compliance", "compliant", "must", "shall",
        "required", "prohibited", "regulation", "regulations",
    ),
    "summarization": (
        "summarize", "summary", "overview", "describe", "outline", "list all", "main points",
    ),
}


class IntentQueryClassifier(QueryClassifier):
    """Domain-independent intent taxonomy (PP-1): label each query
    ``lookup | comparison | aggregation | summarization | troubleshooting | policy`` from
    deterministic keyword cues, so a router dispatches per class — numeric/aggregate questions to a
    structured-data route (once one exists), troubleshooting to a lexical-heavy pipeline, etc. The
    first matching intent in the (consequence-ordered) cue map wins; no cue ⇒ ``lookup``. The matched
    cues are recorded as an annotation. LLM-free, C++-portable; cues are config (``None`` ⇒ built-in)."""

    class Config(QueryClassifier.Config):
        class_name: Literal["intent"] = "intent"
        lookup_type: str = "lookup"  # the no-cue fallback label
        cues: dict[str, list[str]] | None = None  # intent -> cue list; None ⇒ the built-in taxonomy

    config: IntentQueryClassifier.Config

    def classify(self, query: Query, ctx: RetrievalContext) -> None:
        cues = (
            {k: tuple(v) for k, v in self.config.cues.items()}
            if self.config.cues is not None
            else _INTENT_CUES
        )
        text = " ".join(query.text.lower().split())
        words = frozenset(w.strip(string.punctuation) for w in text.split())
        label, matched = self.config.lookup_type, []
        for intent, intent_cues in cues.items():  # first (most consequential) matching intent wins
            hits = [
                c for c in intent_cues if (" " in c and c in text) or (" " not in c and c in words)
            ]
            if hits:
                label, matched = intent, hits
                break
        query.query_type = label
        self._annotate(query, "query_classification", {"label": label, "matched_cues": matched})

