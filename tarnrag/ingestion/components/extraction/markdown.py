"""Markdown extractor — parse Markdown structure into a ``StructuredDocument``.

ATX headings, paragraphs, fenced code, **list items**, and **pipe tables** (→ ``Table``/``TableCell``,
so a markdown table populates ``table_cells`` like Docling's). Each block becomes one element with a
char-offset span into the normalized document text and the header path of its enclosing headings. A
table may follow a blank line or directly interrupt a paragraph (as GitHub renders it); nesting /
multi-line list items coalesce gracefully. Markdown has no row-header syntax, so a table's first column
labels its data rows (``is_row_header`` — the ``table_json`` ``header_cols=1`` convention), which is what
``Table.contextual_text`` and the ``table_lookup`` reasoner resolve rows by.
"""

from __future__ import annotations

import re
from typing import Literal

from tarnrag.contracts import StructuredDocument, Table, TableCell
from tarnrag.ingestion.components.extraction._blocks import Block, assemble
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")  # -, *, +, or "1." / "1)" markers
_FENCE = "```"


class MarkdownExtractor(Extractor):
    """Structure-aware Markdown extraction (headings / paragraphs / code / lists / tables)."""

    class Config(Extractor.Config):
        class_name: Literal["markdown"] = "markdown"

    config: MarkdownExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        elements, text = assemble(self._blocks(self._read_text(source)))
        return self._create_document(source, text, elements, extractor=self.config.class_name, default_kind="markdown")

    @staticmethod
    def _blocks(md: str) -> list[Block]:
        """Split Markdown into ``(kind, normalized_text, level, table)`` blocks; ``level`` is set for
        headings and ``table`` for pipe tables."""
        lines = md.splitlines()
        blocks: list[Block] = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if line.strip().startswith(_FENCE):
                code: list[str] = []
                i += 1
                while i < n and not lines[i].strip().startswith(_FENCE):
                    code.append(lines[i])
                    i += 1
                i += 1  # consume the closing fence (if present)
                blocks.append(("code", "\n".join(code), None, None))
                continue
            heading = _HEADING.match(line)
            if heading:
                blocks.append(("heading", heading.group(2).strip(), len(heading.group(1)), None))
                i += 1
                continue
            if MarkdownExtractor._is_table_start(lines, i):
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < n and "|" in lines[i] and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                table, text = MarkdownExtractor._parse_table(table_lines)
                blocks.append(("table", text, None, table))
                continue
            if _LIST_ITEM.match(line):
                while i < n and (item := _LIST_ITEM.match(lines[i])):
                    blocks.append(("list_item", item.group(1).strip(), None, None))
                    i += 1
                continue
            if not line.strip():
                i += 1
                continue
            para: list[str] = []
            while (
                i < n
                and lines[i].strip()
                and not MarkdownExtractor._is_special(lines[i])
                and not MarkdownExtractor._is_table_start(lines, i)  # a table may interrupt a paragraph
            ):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("paragraph", " ".join(para), None, None))
        return blocks

    @staticmethod
    def _is_special(line: str) -> bool:
        """A line that ends the current paragraph (a heading, fence, or list item starts a new block)."""
        return bool(_HEADING.match(line)) or line.strip().startswith(_FENCE) or bool(_LIST_ITEM.match(line))

    @staticmethod
    def _is_table_start(lines: list[str], i: int) -> bool:
        """Whether ``lines[i]`` is a pipe-table header row (it has a pipe and the next line is a delimiter)."""
        return "|" in lines[i] and i + 1 < len(lines) and MarkdownExtractor._is_delimiter(lines[i + 1])

    @staticmethod
    def _split_row(line: str) -> list[str]:
        """Split a pipe-table row into trimmed cells, dropping the outer pipes."""
        s = line.strip()
        s = s.removeprefix("|")
        s = s.removesuffix("|")
        return [c.strip() for c in s.split("|")]

    @staticmethod
    def _is_delimiter(line: str) -> bool:
        """Whether ``line`` is a pipe-table delimiter row (cells of ``-``, optional leading/trailing ``:``)."""
        cells = MarkdownExtractor._split_row(line)
        return len(cells) > 0 and all(re.fullmatch(r":?-+:?", c) for c in cells)

    @staticmethod
    def _parse_table(lines: list[str]) -> tuple[Table, str]:
        """Parse a pipe table (header, delimiter, data rows) into a ``Table`` + its markdown text."""
        header = MarkdownExtractor._split_row(lines[0])
        data = [MarkdownExtractor._split_row(row) for row in lines[2:]]
        n_cols = len(header)
        cells = [TableCell(id=f"r0c{c}", row=0, col=c, is_column_header=True, text=t) for c, t in enumerate(header)]
        for r, row in enumerate(data, start=1):
            for c in range(n_cols):
                cells.append(
                    TableCell(
                        id=f"r{r}c{c}", row=r, col=c, text=row[c] if c < len(row) else "",
                        is_row_header=c == 0,  # the first column labels the row (no markdown syntax for it)
                    )
                )
        markdown = "\n".join(lines)
        return Table(n_rows=len(data) + 1, n_cols=n_cols, cells=cells, markdown=markdown), markdown
