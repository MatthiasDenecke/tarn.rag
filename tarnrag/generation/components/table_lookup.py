"""The table-lookup reasoner (PP-6) — deterministic numeric answers straight from persisted cells.

The production pattern says: *treat structured data separately — route counts/aggregates to tooling,
not to a reader LLM*. This is that tooling, in-library: a ``Reasoner`` that answers
numeric/aggregate questions from the ``Table`` cells P1 persists (``table_cells``), with **zero LLM
calls**. It retrieves as usual, finds the first hit carrying a real ``Table``, resolves the question
to *(operation, row, year-columns)* by deterministic heuristics, computes, and cites the table
chunk. Anything it cannot resolve confidently is DELEGATED to the wrapped fallback reasoner (or
refused, ``fallback: null`` — the eval posture), never guessed:

- **operation** from cue phrases, most specific first — ``percentage change`` before ``change``
  (every pct-change question contains "change"), and change-cues before ``total``/``sum`` (row
  labels like *"Total (loan)"* would otherwise misfire a sum);
- **columns** are the 4-digit years named in the question (``from 2015 to 2019`` expands the
  range), matched against column-header text and ordered by year value — *later minus earlier* —
  because financial tables usually list the latest year first;
- **row** by token coverage of a row's header label inside the question (best row wins; below the
  coverage floor ⇒ unresolved);
- **values** parse accounting formats: thousands separators, ``$``/``%``, and parenthesized
  negatives ``(2,088)`` → −2088.

Sign conventions are kept honest: change and percentage change are signed
(``(later − earlier) / earlier × 100``), matching the dominant TAT-QA derivation form.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from bausatz import ComponentFactory
from pydantic import Field

from tarnrag.contracts import RetrievalResult, Table
from tarnrag.generation.components.reasoner import (
    ReasonedAnswer,
    ReasonedStep,
    Reasoner,
    SingleHopReasoner,
)
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$")

# Operation cues, MOST SPECIFIC FIRST (see module docstring for the ordering traps).
_OPERATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pct_change", ("percentage change", "% change", "as a percentage of")),
    ("change", ("change in", "change of", "increase", "decrease", "difference between", "difference in")),
    ("average", ("average", "mean")),
    ("sum", ("total", "sum of", "combined")),
)


def _parse_number(text: str) -> float | None:
    """A cell's numeric value: strips ``$``/``%``/thousands separators; ``(x)`` is accounting
    notation for −x; anything non-numeric (labels, em-dashes, blanks) is ``None``."""
    t = text.strip().replace("$", "").replace("%", "").strip()
    if not t or not _NUMBER.match(t.replace(" ", "")):
        return None
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(",", "")
    try:
        value = float(t)
    except ValueError:
        return None
    return -value if negative else value


def _format_number(value: float) -> str:
    """Gold-style rendering: integers bare, else rounded to 2 decimals with trailing zeros trimmed."""
    rounded = round(value, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


class TableLookupReasoner(Reasoner):
    """Answer numeric table questions deterministically from cells; delegate what it can't resolve."""

    class Config(Reasoner.Config):
        class_name: Literal["table_lookup"] = "table_lookup"
        # The reasoner handling everything this one doesn't confidently resolve. ``None`` ⇒ refuse
        # (abstain) instead — the LLM-free posture the numeric eval pins.
        fallback: Reasoner.Config | None = Field(default_factory=SingleHopReasoner.Config)
        top_k: int = 8
        row_coverage: float = 0.5  # min fraction of a row label's tokens that must appear in the question
        refusal: str = "I can't resolve that from the available tables."

    config: TableLookupReasoner.Config

    def __init__(self, config: TableLookupReasoner.Config) -> None:
        super().__init__(config)
        self._fallback: Reasoner | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        self._fallback = (
            factory.build_as(self.config.fallback, Reasoner) if self.config.fallback else None
        )

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        self.ensure_built()
        operation, years = self._operation(query.text), self._years(query.text)
        if operation is None or len(years) < 2:
            return await self._pass_on(query, ctx, evidence=None)
        results = await ctx.retrieval.search(replace(query, top_k=self.config.top_k))
        for index, result in enumerate(results):
            table = result.provenance.table if result.provenance else None
            if table is None:
                continue
            answer = self._resolve(table, query.text, operation, years)
            if answer is not None:
                value, detail = answer
                step = ReasonedStep(claim=f"{detail} = {_format_number(value)}", cited=[index])
                return ReasonedAnswer(answer=_format_number(value), steps=[step], evidence=results)
        return await self._pass_on(query, ctx, evidence=results)

    async def _pass_on(
        self, query: Query, ctx: GenerationContext, evidence: list[RetrievalResult] | None
    ) -> ReasonedAnswer:
        """Delegate to the fallback reasoner, or refuse (abstain) when none is configured."""
        if self._fallback is not None:
            return await self._fallback.reason(query, ctx)
        return ReasonedAnswer(
            answer=self.config.refusal, steps=[], evidence=evidence or [], abstained=True
        )

    # ---------------- question parsing ----------------

    @staticmethod
    def _operation(text: str) -> str | None:
        low = " ".join(text.lower().split())
        for operation, cues in _OPERATIONS:
            if any(cue in low for cue in cues):
                return operation
        return None

    @staticmethod
    def _years(text: str) -> list[int]:
        """The distinct years the question names, ascending; ``from Y1 to Y2`` expands the range
        (an average 'from 2015 to 2019' spans five columns)."""
        years = sorted({int(m.group()) for m in _YEAR.finditer(text)})
        if len(years) == 2 and re.search(rf"from\s+{years[0]}\s+to\s+{years[1]}", text.lower()):
            years = list(range(years[0], years[1] + 1))
        return years

    # ---------------- table resolution ----------------

    def _resolve(
        self, table: Table, question: str, operation: str, years: list[int]
    ) -> tuple[float, str] | None:
        """The computed value + a human-readable derivation, or ``None`` when this table can't
        answer (no matching row, missing year columns, non-numeric cells)."""
        year_cols = self._year_columns(table)
        columns = [year_cols[y] for y in years if y in year_cols]
        if len(columns) != len(years):
            return None
        row, label = self._best_row(table, question)
        if row is None:
            return None
        values = []
        for col in columns:
            cell = table.cell_at(row, col)
            value = _parse_number(cell.text) if cell else None
            if value is None:
                return None
            values.append(value)
        earlier, later = values[0], values[-1]  # years (and hence values) are in ascending order
        if operation == "change":
            return later - earlier, f"change in {label}: {later} − {earlier}"
        if operation == "pct_change":
            if earlier == 0:
                return None
            return (later - earlier) / earlier * 100, f"% change in {label}: ({later} − {earlier}) / {earlier} × 100"
        if operation == "sum":
            return sum(values), f"sum of {label} over {years}"
        if operation == "average":
            return sum(values) / len(values), f"average of {label} over {years}"
        return None  # pragma: no cover - the operation set above is closed (see _OPERATIONS)

    @staticmethod
    def _year_columns(table: Table) -> dict[int, int]:
        """year → column index, from the column-header cells (a header like '2019' or 'FY 2019')."""
        mapping: dict[int, int] = {}
        for cell in table.cells:
            if cell.is_column_header:
                for m in _YEAR.finditer(cell.text):
                    mapping.setdefault(int(m.group()), cell.col)
        return mapping

    def _best_row(self, table: Table, question: str) -> tuple[int | None, str]:
        """The data row whose header label is best covered by the question's tokens (ties → the
        longer, more specific label); ``None`` below the coverage floor."""
        q_tokens = set(re.findall(r"\w+", question.lower()))
        best: tuple[float, int, int, str] = (0.0, 0, -1, "")  # (coverage, label size, row, label)
        labels: dict[int, list[str]] = {}
        for cell in table.cells:
            if cell.is_row_header and cell.text.strip():
                labels.setdefault(cell.row, []).append(cell.text.strip())
        for row, parts in labels.items():
            label = " ".join(parts)
            tokens = set(re.findall(r"\w+", label.lower())) - {"the", "of", "and", "in"}
            if not tokens:
                continue
            coverage = len(tokens & q_tokens) / len(tokens)
            if (coverage, len(tokens)) > (best[0], best[1]):
                best = (coverage, len(tokens), row, label)
        if best[0] >= self.config.row_coverage and best[2] >= 0:
            return best[2], best[3]
        return None, ""
