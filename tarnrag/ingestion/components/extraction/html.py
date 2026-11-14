"""HTML extractor — structure-aware: a ``Source`` → ``StructuredDocument``.

Walks the parsed DOM into reading-order blocks — headings (``<h1..6>`` → levels), paragraphs, list items
(``<li>``), code (``<pre>``), and tables (``<table>`` → ``Table``/``TableCell``, ``<th>`` → header cells)
— then assembles them like the markdown extractor (shared ``assemble``). ``beautifulsoup4`` is the
optional ``parsers`` extra, imported lazily.
"""

from __future__ import annotations

from typing import Literal

from tarnrag.contracts import StructuredDocument, Table, TableCell
from tarnrag.ingestion.components.extraction._blocks import Block, assemble
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP = {"script", "style", "head"}  # non-content subtrees


class HtmlExtractor(Extractor):
    """Structure-aware HTML extraction (headings / paragraphs / lists / code / tables)."""

    class Config(Extractor.Config):
        class_name: Literal["html"] = "html"

    config: HtmlExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self._read_text(source), "html.parser")
        blocks: list[Block] = []
        self._walk(soup.body or soup, blocks)  # a full doc has <body>; a fragment is walked whole
        elements, text = assemble(blocks)
        return self._create_document(source, text, elements, extractor=self.config.class_name, default_kind="html")

    @staticmethod
    def _walk(node: object, blocks: list[Block]) -> None:
        """Depth-first DOM walk: emit a block for each known structural tag (and don't recurse into it);
        recurse into containers (``<div>``/``<section>``/…); capture loose text as paragraphs."""
        from bs4 import NavigableString

        for child in node.children:  # type: ignore[attr-defined]
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    blocks.append(("paragraph", text, None, None))
                continue
            name = child.name
            if name in _SKIP:
                continue
            if name in _HEADINGS:
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(("heading", text, int(name[1]), None))
            elif name == "p":
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(("paragraph", text, None, None))
            elif name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        blocks.append(("list_item", text, None, None))
            elif name == "table":
                table, markdown = HtmlExtractor._table(child)
                if table.cells:
                    blocks.append(("table", markdown, None, table))
            elif name == "pre":
                text = child.get_text().strip("\n")
                if text.strip():
                    blocks.append(("code", text, None, None))
            else:
                HtmlExtractor._walk(child, blocks)

    @staticmethod
    def _table(table_tag: object) -> tuple[Table, str]:
        """Parse a ``<table>`` into a ``Table`` + a markdown rendering (the table chunk's searchable text).

        Cells are placed on the real grid: a ``colspan`` pushes the cells after it right, and a
        ``rowspan`` reserves its column in the rows below — so each cell's ``col`` lines up with the
        header above it (counting ``<td>``s per row would shift everything after a span). Header flags:
        a ``<th>`` in the first row is a column header; in a later row, the first column (``<td>`` or
        ``<th>``) and any other ``<th>`` label the row. Rows of nested tables belong to those tables."""
        rows = [tr for tr in table_tag.find_all("tr") if tr.find_parent("table") is table_tag]  # type: ignore[attr-defined]
        occupied: set[tuple[int, int]] = set()  # grid slots taken by a cell (incl. spans from rows above)
        placed: list[tuple[int, int, str]] = []
        cells: list[TableCell] = []
        for r, tr in enumerate(rows):
            c = 0
            for cell in tr.find_all(["td", "th"], recursive=False):
                while (r, c) in occupied:
                    c += 1
                row_span = min(HtmlExtractor._span(cell.get("rowspan")), len(rows) - r)
                col_span = HtmlExtractor._span(cell.get("colspan"))
                occupied.update((r + dr, c + dc) for dr in range(row_span) for dc in range(col_span))
                text = cell.get_text(" ", strip=True)
                is_th = cell.name == "th"
                is_column_header = is_th and r == 0
                cells.append(
                    TableCell(
                        id=f"r{r}c{c}", row=r, col=c, text=text,
                        is_column_header=is_column_header,
                        is_row_header=not is_column_header and r > 0 and (c == 0 or is_th),
                        row_span=row_span, col_span=col_span,
                    )
                )
                placed.append((r, c, text))
                c += col_span
        n_cols = max((col for _, col in occupied), default=-1) + 1
        grid = [[""] * n_cols for _ in rows]
        for r, c, text in placed:
            grid[r][c] = text
        markdown = HtmlExtractor._render(grid)
        return Table(n_rows=len(rows), n_cols=n_cols, cells=cells, markdown=markdown), markdown

    @staticmethod
    def _span(value: object) -> int:
        """A ``rowspan`` / ``colspan`` attribute as a positive int (absent / malformed / 0 → 1)."""
        try:
            return max(1, int(str(value).strip())) if value is not None else 1
        except ValueError:
            return 1

    @staticmethod
    def _render(grid: list[list[str]]) -> str:
        """Render a cell grid as a markdown table (header = row 0)."""
        if not grid:
            return ""
        n_cols = max(len(row) for row in grid)

        def line(cells: list[str]) -> str:
            return "| " + " | ".join(cells + [""] * (n_cols - len(cells))) + " |"

        out = [line(grid[0]), "| " + " | ".join(["---"] * n_cols) + " |"]
        out += [line(row) for row in grid[1:]]
        return "\n".join(out)
