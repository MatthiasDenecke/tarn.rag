"""CleanAndNormalizeStage — strip control characters and normalize whitespace.

A document-phase stage: after ``LoadAndParse`` an item carries a ``StructuredDocument`` — the substrate
the chunker splits and every ``Span`` indexes into — so cleaning only ``item.content`` would change
nothing downstream. The stage therefore cleans the **document text** and remaps every char offset into it
(element geometry, sentence spans, annotation spans, table-cell geometry) through the edit, so
``doc.text[span] == element.text`` still holds afterwards and highlights stay exact. Code blocks and
tables keep their whitespace (indentation / the grid rendering of the cells is structure there, owned by
the extractor); line endings and control characters are normalized everywhere. An item without a
document (a plain-content pipeline, or one that cleans after chunking) has just its text cleaned.

The rules, applied in order (each a regex edit, composed through one offset map):

1. ``\\r\\n`` / lone ``\\r`` → ``\\n``;
2. C0 control characters (except tab / newline) and NUL → removed;
3. with ``collapse_whitespace``: a run of spaces/tabs → one space, and 3+ newlines → a blank line;
4. leading / trailing whitespace of the whole text → removed.

Rules 3–4 never edit inside a code block or a table.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator
from typing import Any, Literal

from tarnrag.contracts import ElementKind, PipelineItem, Span, StructuredDocument
from tarnrag.core.hashing import compute_content_hash
from tarnrag.ingestion.pipeline.pipeline import MapperStage

_NEWLINES = (re.compile(r"\r\n?"), "\n")
_CONTROL = (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"), "")
_BLANKS = (re.compile(r"[ \t]{2,}|\t"), " ")
_PARAGRAPHS = (re.compile(r"\n{3,}"), "\n\n")
_EDGES = (re.compile(r"\A\s+|\s+\Z"), "")

# Elements whose whitespace is structure (indentation, the grid layout) — exempt from rules 3–4.
_VERBATIM = frozenset({ElementKind.CODE, ElementKind.TABLE})

_Edit = tuple[int, int, str]  # (start, end, replacement) over the text BEFORE the edit pass


class _OffsetMap:
    """Maps char offsets in the text before one edit pass to offsets after it. A boundary inside an
    edited region maps into its replacement (clamped), so spans stay ordered and in range."""

    def __init__(self, edits: list[_Edit]) -> None:
        self._starts = [s for s, _, _ in edits]
        self._edits = edits
        self._shift_before: list[int] = []  # net length change of all edits before edit i
        shift = 0
        for s, e, r in edits:
            self._shift_before.append(shift)
            shift += len(r) - (e - s)

    def __call__(self, pos: int) -> int:
        i = bisect_right(self._starts, pos) - 1  # the last edit starting at or before pos
        if i < 0:
            return pos
        s, e, r = self._edits[i]
        shift = self._shift_before[i]
        if pos >= e:  # past this edit: shifted by everything up to and including it
            return pos + shift + len(r) - (e - s)
        return s + shift + min(pos - s, len(r))  # inside it: into the replacement


class _Protected:
    """The protected ``(start, end)`` ranges, sorted and merged, with an O(log n) overlap test — a
    whitespace match is checked against them once per match, and a table-heavy filing has hundreds."""

    def __init__(self, ranges: list[tuple[int, int]]) -> None:
        merged: list[list[int]] = []
        for s, e in sorted(ranges):
            if merged and s < merged[-1][1]:  # overlapping: one range (touching ones stay apart — exactness)
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self._starts = [s for s, _ in merged]
        self._ends = [e for _, e in merged]  # non-decreasing too: each range starts where the last ended, or later

    def overlaps(self, start: int, end: int) -> bool:
        """Whether ``[start, end)`` intersects a protected range — exactly ``any(s < end and start < e)``
        over the original ranges."""
        i = bisect_right(self._ends, start)  # the first range ending after ``start``; later ones begin later
        return i < len(self._starts) and self._starts[i] < end

    def remapped(self, mapping: _OffsetMap) -> _Protected:
        """The ranges carried through an edit pass (the map is monotone, so order and disjointness hold)."""
        return _Protected([(mapping(s), mapping(e)) for s, e in zip(self._starts, self._ends)])


def _clean(
    text: str, collapse: bool, protected: list[tuple[int, int]] = ()
) -> tuple[str, list[_OffsetMap]]:
    """Clean ``text`` by the module's rules, never collapsing whitespace inside a ``protected`` range
    (``(start, end)`` offsets into ``text``). Returns the cleaned text and the per-pass offset maps
    (compose them, in order, to carry an offset from ``text`` into the result)."""
    rules = [(_NEWLINES, False), (_CONTROL, False)]
    if collapse:
        rules += [(_BLANKS, True), (_PARAGRAPHS, True)]
    rules.append((_EDGES, True))
    maps: list[_OffsetMap] = []
    guard = _Protected(list(protected))
    for (pattern, repl), respects_protection in rules:
        edits = [
            (m.start(), m.end(), repl)
            for m in pattern.finditer(text)
            if m.group() != repl and not (respects_protection and guard.overlaps(m.start(), m.end()))
        ]
        if not edits:
            continue
        mapping = _OffsetMap(edits)
        parts, last = [], 0
        for s, e, r in edits:
            parts += [text[last:s], r]
            last = e
        text = "".join(parts) + text[last:]
        guard = guard.remapped(mapping)
        maps.append(mapping)
    return text, maps


def clean_text(text: str, *, collapse_whitespace: bool = True, preserve_whitespace: bool = False) -> str:
    """The cleaned form of a standalone string (no offsets to carry) — the plain-text path.
    ``preserve_whitespace`` applies only rules 1–2 (line endings, control characters): the treatment a
    code block or a table gets inside a document."""
    if preserve_whitespace:
        return _clean(text, False, [(0, len(text))])[0]
    return _clean(text, collapse_whitespace)[0]


class CleanAndNormalizeStage(MapperStage):
    """
    Strip control characters and (optionally) collapse runs of whitespace before chunking — over the
    structured document when there is one (offsets remapped), else over the item's text.
    """

    class Config(MapperStage.Config):
        class_name: Literal["CleanAndNormalize"] = "CleanAndNormalize"
        collapse_whitespace: bool = True

    config: CleanAndNormalizeStage.Config

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return clean_text(text, collapse_whitespace=self.config.collapse_whitespace), {"cleaned": True}

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        if item.document is None:
            yield from super().process(item)  # plain text (or a chunk-phase item): just the string
            return
        document = self._clean_document(item.document)
        yield item.derive(
            content=document.text, metadata={**item.metadata, "cleaned": True}, document=document
        )

    def _clean_document(self, source: StructuredDocument) -> StructuredDocument:
        """A cleaned copy of ``source`` (the upstream document is never mutated) with every span
        remapped into the cleaned text.

        An element's text is **re-sliced** from the cleaned text when it was exactly its one span's slice
        (every built-in extractor's elements), so ``doc.text[span] == element.text`` still holds. Any
        other element — several spans, or text that differs from its span — has its text cleaned **on its
        own** by the same rules: its spans still index the cleaned text exactly, but its text is a separate
        rendering and can differ from the slices at a span's edge (a stripped edge, a whitespace run that
        straddles the span boundary)."""
        collapse = self.config.collapse_whitespace
        doc = source.model_copy(deep=True)
        protected = [
            (span.start, span.end)
            for el in doc.elements
            if el.kind in _VERBATIM
            for span in el.geometry
        ]
        text, maps = _clean(source.text, collapse, protected)

        def remap(span: Span) -> Span:
            start, end = span.start, span.end
            for mapping in maps:
                start, end = mapping(start), mapping(end)
            return span.model_copy(update={"start": start, "end": end})

        def remap_all(spans: list[Span] | None) -> list[Span] | None:
            return None if spans is None else [remap(s) for s in spans]

        for el in doc.elements:
            exact = len(el.geometry) == 1 and source.text[el.geometry[0].start : el.geometry[0].end] == el.text
            el.geometry = remap_all(el.geometry)
            el.sentences = remap_all(el.sentences)
            for annotation in el.annotations:
                annotation.span = remap_all(annotation.span)
            verbatim = el.kind in _VERBATIM
            old_text = el.text
            el.text = (
                text[el.geometry[0].start : el.geometry[0].end]
                if exact
                else clean_text(el.text, collapse_whitespace=collapse, preserve_whitespace=verbatim)
            )
            if el.table is not None:  # cells + markdown are the table's own renderings — whitespace kept
                for cell in el.table.cells:
                    cell.text = clean_text(cell.text, preserve_whitespace=True)
                    cell.geometry = remap_all(cell.geometry)
                el.table.markdown = (
                    el.text
                    if el.table.markdown == old_text
                    else clean_text(el.table.markdown, preserve_whitespace=True)
                )
        for annotation in doc.annotations:
            annotation.span = remap_all(annotation.span)
        doc.text = text
        doc.content_hash = compute_content_hash(text)
        return doc
