"""Deterministic query-text cues — the exact-match signals shared across layers.

The structural query classifier reads these cues to label lexical intent (a quoted span or an
identifier ⇒ ``query_type='lexical'``), and the sparse-query builders act on the same cues (a
quoted span or a punctuation-split identifier becomes a required phrase). One definition here
keeps the two from drifting: what the classifier calls an identifier is exactly what the FTS
builder phrases. LLM-free, deterministic, C++-portable.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

_QUOTED = re.compile(r'"([^"]+)"')
_VERSION = re.compile(r"\d+(?:[.\-]\d+)+")  # 6.4, 1.2.3, 2024-01 — dotted/hyphenated number runs


def quoted_spans(text: str) -> list[str]:
    """The double-quoted spans in ``text`` — explicit exact-match intent — quote-free, blank-free."""
    return [s for s in (m.strip() for m in _QUOTED.findall(text)) if s]


def looks_like_identifier(token: str) -> bool:
    """Whether a whitespace token reads as an identifier / code (a language-independent exact-match
    cue): a ``§`` reference, a letter+digit mix (``BM25``, ``L6``, ``v2``), an all-caps acronym
    (``API``, ``PDF``), or a dotted/hyphenated number (``6.4``, ``1.2.3``)."""
    if "§" in token:
        return True
    core = token.strip(string.punctuation)
    if not core:
        return False
    has_alpha = any(c.isalpha() for c in core)
    has_digit = any(c.isdigit() for c in core)
    if has_alpha and has_digit:
        return True
    if len(core) >= 2 and core.isupper() and has_alpha:
        return True
    return bool(_VERSION.fullmatch(core))


_QUOTED_SPAN = re.compile(r'"[^"]*"')  # consumed quoted spans, struck from the OR remainder
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class LexicalQuery:
    """A query text decomposed by its exact-match cues, for the dialects' sparse-query builders to render
    the same way: ``phrases`` must each match as an adjacent phrase (a double-quoted span, or an
    identifier whose punctuation splits it — ``6.4.2``, ``2024-01``, ``§6.4`` — whose pieces OR-ed would
    match scattered anywhere); ``terms`` are the recall-first remainder, any of which may match.

    ``phrases`` keep their raw text (each dialect tokenizes them its own way — FTS5 needs the word
    pieces, Postgres' parser reads ``6.4.2`` as one version lexeme on both index and query side);
    ``terms`` are lower-cased word pieces, deduplicated, in order."""

    phrases: tuple[str, ...]
    terms: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.phrases or self.terms)


def lexical_query(text: str) -> LexicalQuery:
    """Split ``text`` into required phrases + OR terms (see :class:`LexicalQuery`) — the one parse both
    sparse backends share, so SQLite (FTS5) and Postgres (tsquery) agree on what a query means."""
    phrases = [span for span in quoted_spans(text) if _WORD.search(span)]
    terms: list[str] = []
    for token in _QUOTED_SPAN.sub(" ", text).split():
        pieces = _WORD.findall(token.lower())
        if len(pieces) > 1 and looks_like_identifier(token):
            phrases.append(token)  # a split identifier: adjacency is the meaning
        else:
            for piece in pieces:  # dedupe one at a time — a token like 20,000,000 repeats pieces
                if piece not in terms:
                    terms.append(piece)
    return LexicalQuery(tuple(phrases), tuple(terms))
