"""The shared query-text cues (``tarnrag.core.text``) — here, the lexical parse both sparse backends render."""

from tarnrag.core.text import lexical_query


def test_lexical_query_splits_required_phrases_from_or_terms():
    """Quoted spans + split identifiers are required phrases (raw text — each dialect tokenizes them its
    own way: FTS5 by word pieces, Postgres' parser as one version lexeme); the rest are deduplicated
    lower-case OR terms."""
    q = lexical_query('check "goodwill impairment" per §6.4 and 6.4.2, 20,000,000 total')
    assert q.phrases == ("goodwill impairment", "§6.4", "6.4.2,")
    assert q.terms == ("check", "per", "and", "20", "000", "total")
    assert not lexical_query('!!! " " ...')  # nothing usable
    assert lexical_query("BM25 ranking").terms == ("bm25", "ranking")  # a one-piece identifier is a term
