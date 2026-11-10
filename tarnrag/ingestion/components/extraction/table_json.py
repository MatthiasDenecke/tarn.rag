"""Structured-table extractor — a JSON grid becomes a native ``Table`` element (P1).

For callers that already hold a parsed table (a benchmark loader, an API client, a spreadsheet
exporter), this is the **native table ingest path**: the document's content is a JSON grid
(``[["", "2019", "2018"], ["Goodwill", "1,910", "2,130"], …]``), and it extracts to a single
atomic ``TABLE`` element carrying a full ``Table`` — cells with grid positions and header flags —
so the chunker's table leaf persists ``table_cells``, retrieval can cite cells, and the Embed
stage's ``contextualize_tables`` rendering has structure to work from. The element ``text`` (what
is stored, FTS-indexed, and — without the embed flag — embedded) is the `` | ``-joined grid, the
exact cell tokens.

Header layout is config (``header_rows`` / ``header_cols``, defaults 1/1 — the common
first-row-column-headers, first-column-row-labels shape); the content may also be an object
``{"grid": …, "header_rows": …, "header_cols": …}`` to override per document.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from tarnrag.contracts import (
    Element,
    ElementKind,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source


class TableJsonExtractor(Extractor):
    """A JSON grid → one atomic ``TABLE`` element with a fully-celled ``Table``."""

    class Config(Extractor.Config):
        class_name: Literal["table_json"] = "table_json"
        header_rows: int = Field(default=1, ge=0)  # leading rows that are column headers
        header_cols: int = Field(default=1, ge=0)  # leading columns that are row labels

    config: TableJsonExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        grid, header_rows, header_cols = self._parse(self._read_text(source))
        table = self._table(grid, header_rows, header_cols)
        text = table.markdown
        elements = (
            [
                Element(
                    id="t0",
                    kind=ElementKind.TABLE,
                    text=text,
                    geometry=[Span(start=0, end=len(text))],
                    table=table,
                )
            ]
            if grid
            else []
        )
        return self._create_document(
            source, text, elements, extractor=self.config.class_name, default_kind="table"
        )

    def _parse(self, content: str) -> tuple[list[list[str]], int, int]:
        """The grid + header shape from the JSON content — a bare array, or an object with per-document
        ``header_rows`` / ``header_cols`` overriding the config."""
        data = json.loads(content)
        if isinstance(data, dict):
            grid = data.get("grid")
            header_rows = int(data.get("header_rows", self.config.header_rows))
            header_cols = int(data.get("header_cols", self.config.header_cols))
        else:
            grid, header_rows, header_cols = data, self.config.header_rows, self.config.header_cols
        if not isinstance(grid, list) or any(not isinstance(row, list) for row in grid):
            raise ValueError("table_json content must be a JSON grid (list of rows) or {'grid': …}")
        return [[str(c) for c in row] for row in grid], header_rows, header_cols

    @staticmethod
    def _table(grid: list[list[str]], header_rows: int, header_cols: int) -> Table:
        """Cells with grid positions + header flags; ``markdown`` is the `` | ``-joined grid (the
        exact cell tokens, row per line — the display / BM25 form)."""
        cells = [
            TableCell(
                id=f"r{r}c{c}",
                row=r,
                col=c,
                text=text,
                is_column_header=r < header_rows,
                is_row_header=c < header_cols and r >= header_rows,
            )
            for r, row in enumerate(grid)
            for c, text in enumerate(row)
        ]
        return Table(
            n_rows=len(grid),
            n_cols=max((len(row) for row in grid), default=0),
            cells=cells,
            markdown="\n".join(" | ".join(row) for row in grid),
        )
