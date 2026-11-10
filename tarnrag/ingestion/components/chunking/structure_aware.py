"""Structure-aware chunker — split on the document's structure, build the auto-merging tree.

Walks the ``StructuredDocument`` elements into a **section tree** by heading depth (h1 ⊃ h2 ⊃ h3 …),
then emits chunks bottom-up:

- **leaves** (``level`` 0) — a section's own body packed under ``max_chars``; a **table is its own
  atomic leaf** (never split, never merged), carrying its ``Table``;
- **section parents** (``level`` > 0) — a chunk over a whole section, emitted only where a section
  groups **≥ 2 children** (its leaves + its subsections). ``level`` = height above the leaves, so
  nested subsections produce ``level`` 2, 3, … — the multi-level auto-merging tree.

Degenerate single-child chains collapse (a heading that wraps exactly one child adds no grouping, so no
parent is emitted — the child carries the deeper ``header_path``). The document root is **not** wrapped
in a parent: top-level sections are a forest, so there is never a single whole-document chunk. Each
chunk carries its section ``header_path``, source element ids, and aggregated geometry (page boxes
included).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from tarnrag.contracts import (
    Annotation,
    ChunkProvenance,
    Element,
    ElementKind,
    Span,
    StructuredDocument,
    Table,
)
from tarnrag.ingestion.components.chunking.chunker import Chunker

_JOIN = "\n\n"  # joins packed element texts within a chunk


@dataclass
class _Section:
    """A node of the heading tree: a heading (None for the document root), its direct body elements
    (before any deeper heading), and its subsections."""

    heading: Element | None
    body: list[Element] = field(default_factory=list)
    children: list[_Section] = field(default_factory=list)


@dataclass
class _Node:
    """A node of the *chunk* tree: an emitted chunk and the child chunks that point at it as parent."""

    chunk: Chunker.TempChunk
    children: list[_Node] = field(default_factory=list)


class StructureAwareChunker(Chunker):
    """Split on document structure (sections / tables) into a multi-level leaf + section-parent tree."""

    class Config(Chunker.Config):
        class_name: Literal["structure_aware"] = "structure_aware"
        max_chars: int = Field(default=1200, gt=0)  # soft budget for packing text elements into a leaf

    config: StructureAwareChunker.Config

    def chunk(self, document: StructuredDocument) -> list[Chunker.TempChunk]:
        root = self._build_tree(document.elements)
        # The root's own body (preamble) becomes top-level leaves; each top section becomes a tree.
        forest = [_Node(leaf) for leaf in self._leaves(root.body, [])]
        forest += [node for sub in root.children if (node := self._emit_section(sub)) is not None]
        chunks: list[Chunker.TempChunk] = []
        for node in forest:  # pre-order: a parent is emitted before its children (parent_index points back)
            self._flatten(node, chunks, None)
        if not chunks and document.text.strip():  # no usable structure → one leaf over the whole text
            chunks.append(
                self._create_temp_chunk(
                    document.text,
                    element_ids=[e.id for e in document.elements],
                    geometry=[Span(start=0, end=len(document.text))],
                    header_path=[],
                )
            )
        return chunks

    @staticmethod
    def _build_tree(elements: list[Element]) -> _Section:
        """Parse the flat element list into a heading tree (a stack keyed by heading ``level``). A
        non-heading element attaches to the deepest open heading (its enclosing section)."""
        root = _Section(heading=None)
        stack: list[tuple[int, _Section]] = [(0, root)]  # (heading level, section); root at level 0
        for el in elements:
            if el.kind == ElementKind.HEADING:
                level = el.level if el.level is not None else 1
                while len(stack) > 1 and stack[-1][0] >= level:  # close siblings / deeper sections
                    stack.pop()
                section = _Section(heading=el)
                stack[-1][1].children.append(section)
                stack.append((level, section))
            else:
                stack[-1][1].body.append(el)
        return root

    def _emit_section(self, section: _Section) -> _Node | None:
        """Build the chunk subtree for one section: its body leaves + its subsections' nodes. Emit a
        parent only when there are ≥2 children; collapse a single child (no grouping value); a
        childless (heading-only) section emits nothing — it survives in its descendants' header_path."""
        heading = section.heading
        section_path = (heading.header_path + [heading.text]) if heading else []
        children = [_Node(leaf) for leaf in self._leaves(section.body, section_path)]
        children += [node for sub in section.children if (node := self._emit_section(sub)) is not None]
        if not children:
            return None
        if len(children) == 1:
            return children[0]  # collapse: this heading groups one child — pass it up unwrapped
        return _Node(self._parent(heading, section_path, [c.chunk for c in children]), children)

    def _flatten(self, node: _Node, chunks: list[Chunker.TempChunk], parent_index: int | None) -> None:
        """Append the subtree in pre-order, stamping each chunk's ``parent_index`` (its parent's slot)."""
        node.chunk.parent_index = parent_index
        my_index = len(chunks)
        chunks.append(node.chunk)
        for child in node.children:
            self._flatten(child, chunks, my_index)

    def _leaves(self, body: list[Element], section_path: list[str]) -> list[Chunker.TempChunk]:
        """Pack a section's body into leaves: tables stay atomic; text/code packs under ``max_chars``."""
        leaves: list[Chunker.TempChunk] = []
        buf_text: list[str] = []
        buf_els: list[Element] = []
        size = 0

        def flush() -> None:
            nonlocal buf_text, buf_els, size
            if buf_els:
                leaves.append(self._leaf(_JOIN.join(buf_text), buf_els, section_path))
                buf_text, buf_els, size = [], [], 0

        for el in body:
            if el.kind == ElementKind.TABLE:
                flush()
                leaves.append(self._table_leaf(el, section_path))
                continue
            if not el.text:
                continue
            if len(el.text) > self.config.max_chars:  # oversize single element → character-split it
                flush()
                leaves.extend(self._oversize(el, section_path))
                continue
            if buf_els and size + len(el.text) > self.config.max_chars:
                flush()
            buf_text.append(el.text)
            buf_els.append(el)
            size += len(el.text)
        flush()
        return leaves

    def _create_temp_chunk(
        self,
        text: str,
        *,
        element_ids: list[str],
        geometry: list[Span],
        header_path: list[str],
        annotations: list[Annotation] | None = None,
        level: int = 0,
        table: Table | None = None,
    ) -> Chunker.TempChunk:
        """Build a ``TempChunk`` + its ``ChunkProvenance`` — the one constructor the leaf / table /
        oversize / parent / fallback paths share (each supplies its source elements, geometry, and
        header path; the content hash is always computed here)."""
        return Chunker.TempChunk(
            text=text,
            provenance=ChunkProvenance(
                source_element_ids=element_ids,
                geometry=geometry,
                header_path=header_path,
                content_hash=self._hash(text),
                level=level,
                annotations=annotations or [],
                table=table,
            ),
        )

    def _leaf(self, text: str, elements: list[Element], header_path: list[str]) -> Chunker.TempChunk:
        """A level-0 leaf over ``elements``, packed into ``text``."""
        return self._create_temp_chunk(
            text,
            element_ids=[e.id for e in elements],
            geometry=self._union_geometry(elements),
            header_path=header_path,
            annotations=self._annotations(elements),
        )

    def _table_leaf(self, element: Element, header_path: list[str]) -> Chunker.TempChunk:
        """A table's atomic leaf — its markdown is the searchable text; the ``Table`` rides on the
        provenance so cell geometry + header addressing can be persisted (the table_cells slice)."""
        text = element.text or (element.table.markdown if element.table else "")
        return self._create_temp_chunk(
            text,
            element_ids=[element.id],
            geometry=self._union_geometry([element]),
            header_path=header_path,
            annotations=self._annotations([element]),
            table=element.table,
        )

    def _oversize(self, element: Element, header_path: list[str]) -> list[Chunker.TempChunk]:
        """Character-split a single element that alone exceeds ``max_chars`` (e.g. a huge paragraph).
        The pieces share the element's geometry — box-precise sub-spans are a later refinement."""
        size = self.config.max_chars
        text = element.text
        return [
            self._create_temp_chunk(
                text[i : i + size],
                element_ids=[element.id],
                geometry=self._union_geometry([element]),
                header_path=header_path,
                annotations=self._annotations([element]),
            )
            for i in range(0, len(text), size)
        ]

    def _parent(
        self, heading: Element | None, header_path: list[str], children: list[Chunker.TempChunk]
    ) -> Chunker.TempChunk:
        """A section parent over its ``children`` — text + provenance aggregated across them (and the
        heading), ``level`` one above the deepest child: the auto-merge target for the whole section."""
        head_ids = [heading.id] if heading else []
        head_geometry = list(heading.geometry) if heading else []
        head_annotations = list(heading.annotations) if heading else []
        texts = ([heading.text] if heading else []) + [c.text for c in children]
        text = _JOIN.join(t for t in texts if t)
        return self._create_temp_chunk(
            text,
            element_ids=head_ids + [eid for c in children for eid in c.provenance.source_element_ids],
            # INVARIANT (relied on by retrieval's ``AutoMerger._drop_covered``): a parent's geometry is the
            # literal concatenation of its heading's and its children's spans — every descendant's
            # ``(start, end)`` appears in it verbatim. Coalescing adjacent spans here would silently stop the
            # merger from recognizing a section's own fragments (pinned by test_chunking).
            geometry=head_geometry + [s for c in children for s in c.provenance.geometry],
            header_path=header_path,
            level=1 + max(c.provenance.level for c in children),
            annotations=head_annotations + [a for c in children for a in c.provenance.annotations],
        )
