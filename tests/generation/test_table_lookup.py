"""The table_lookup reasoner (PP-6): deterministic numeric answers from persisted cells — parsing,
operation/row/column resolution, sign conventions, and the delegate-or-abstain contract."""

import json

from bausatz import ComponentFactory

from tarnrag.contracts import ChunkProvenance, RetrievalResult, Table, TableCell
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import GenerationContext, TableLookupReasoner
from tarnrag.generation.components.reasoner import Reasoner
from tarnrag.generation.components.table_lookup import _format_number, _parse_number
from tarnrag.retrieval.types import Query


def test_parse_number_handles_accounting_formats():
    assert _parse_number("14,740") == 14740
    assert _parse_number("(2,088)") == -2088  # parenthesized negative
    assert _parse_number("$828.8") == 828.8
    assert _parse_number("17.2%") == 17.2
    assert _parse_number("-7") == -7
    assert _parse_number("—") is None and _parse_number("") is None and _parse_number("n/a") is None


def test_format_number_matches_gold_style():
    assert _format_number(16650.0) == "16650"  # integers bare
    assert _format_number(-56.5) == "-56.5"
    assert _format_number(-87.04206) == "-87.04"  # 2 decimals, trailing zeros trimmed


def _lookup(**cfg) -> TableLookupReasoner:
    return ComponentFactory.get().create_as(
        {"class_name": "table_lookup", "fallback": None, **cfg}, TableLookupReasoner
    )


def test_operation_cue_ordering():
    op = TableLookupReasoner._operation
    assert op("What was the percentage change in Total (loan) in 2019 from 2018?") == "pct_change"
    assert op("What was the change in Total (loan) in 2019 from 2018?") == "change"  # not 'sum'!
    assert op("What is the total goodwill impairment in 2018 and 2019?") == "sum"
    assert op("How much was the average operating income from 2015 to 2019?") == "average"
    assert op("What was the goodwill impairment in 2019?") is None  # a span question — not ours


def test_year_range_expansion():
    years = TableLookupReasoner._years
    assert years("between 2018 and 2019") == [2018, 2019]
    assert years("from 2015 to 2019") == [2015, 2016, 2017, 2018, 2019]  # a 5-column average
    assert years("in 2019") == [2019]


# The fixture table lists the LATEST year first — the common financial layout — so later-minus-
# earlier must come from the year values, never the column order.
def _goodwill_table() -> Table:
    return Table(
        n_rows=2, n_cols=3,
        cells=[
            TableCell(id="h1", row=0, col=1, is_column_header=True, text="2019"),
            TableCell(id="h2", row=0, col=2, is_column_header=True, text="2018"),
            TableCell(id="r1", row=1, col=0, is_row_header=True, text="Goodwill impairment"),
            TableCell(id="v1", row=1, col=1, text="1,910"),
            TableCell(id="v2", row=1, col=2, text="14,740"),
        ],
    )


def _table_hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="t", text="grid", score=1.0, component_scores={}, document_id="d",
        source_kind="table", standard_id=None, locator=None, license_class="public_domain",
        provenance=ChunkProvenance(content_hash="h", table=_goodwill_table()),
    )


class _FakeRetrieval:
    def __init__(self, results):
        self._results = list(results)

    async def search(self, query):
        return list(self._results)


async def _answer(question: str) -> object:
    ctx = GenerationContext(_FakeRetrieval([_table_hit()]), llm=None)  # no LLM anywhere
    return await _lookup().reason(Query(text=question), ctx)


async def test_change_is_later_minus_earlier_despite_column_order():
    out = await _answer("What was the change in goodwill impairment between 2018 and 2019?")
    assert out.answer == "-12830"  # 1,910 (2019) − 14,740 (2018): signed, by year not column order
    assert out.steps[0].cited == [0]  # the computation cites the table hit


async def test_percentage_change_is_signed():
    out = await _answer("What was the percentage change in goodwill impairment between 2018 and 2019?")
    assert out.answer == "-87.04"  # (1910 − 14740) / 14740 × 100


async def test_sum_and_average():
    out = await _answer("What is the total goodwill impairment in 2018 and 2019?")
    assert out.answer == "16650"
    out = await _answer("What is the average goodwill impairment in 2018 and 2019?")
    assert out.answer == "8325"


async def test_unresolvable_row_abstains_instead_of_guessing():
    out = await _answer("What was the change in operating revenue between 2018 and 2019?")
    assert out.abstained is True and out.steps == []  # no matching row — refuse, never guess


async def test_non_numeric_question_delegates_to_the_fallback():
    calls = {"n": 0}

    def reply(prompt):
        calls["n"] += 1
        return json.dumps({"answer": "a definition", "steps": []})

    reasoner = ComponentFactory.get().create_as(
        {"class_name": "table_lookup", "fallback": {"class_name": "single_hop"}}, TableLookupReasoner
    )
    ctx = GenerationContext(_FakeRetrieval([_table_hit()]), StaticLanguageModel(reply))
    out = await reasoner.reason(Query(text="What is goodwill impairment?"), ctx)
    assert out.answer == "a definition" and calls["n"] == 1  # the reader handled it


async def test_registers_as_a_reasoner():
    got = ComponentFactory.get().create_as({"class_name": "table_lookup"}, Reasoner)
    assert isinstance(got, TableLookupReasoner)  # composes anywhere a reasoner spec goes


def test_parse_number_rejects_regexy_but_unfloatable_input():
    assert _parse_number("(,)") is None  # passes the shape regex, fails float()


def _mixed_table() -> Table:
    # A second row whose label is all stopwords (skipped), a dash-valued cell, and only 2019/2018.
    return Table(
        n_rows=4, n_cols=3,
        cells=[
            TableCell(id="h1", row=0, col=1, is_column_header=True, text="2019"),
            TableCell(id="h2", row=0, col=2, is_column_header=True, text="2018"),
            TableCell(id="r1", row=1, col=0, is_row_header=True, text="of the"),
            TableCell(id="r2", row=2, col=0, is_row_header=True, text="Deferred revenue"),
            TableCell(id="v1", row=2, col=1, text="—"),
            TableCell(id="v2", row=2, col=2, text="29"),
            # A row that LOSES the best-row contest (zero question coverage).
            TableCell(id="r3", row=3, col=0, is_row_header=True, text="Net cash used"),
            TableCell(id="v3", row=3, col=1, text="1"),
            TableCell(id="v4", row=3, col=2, text="2"),
        ],
    )


def _hit_with(table: Table, chunk_id: str = "t2") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, text="grid", score=1.0, component_scores={}, document_id="d",
        source_kind="table", standard_id=None, locator=None, license_class="public_domain",
        provenance=ChunkProvenance(content_hash="h", table=table),
    )


def _text_hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="p", text="a paragraph without any table", score=1.0, component_scores={},
        document_id="d", source_kind="text", standard_id=None, locator=None,
        license_class="public_domain",
    )


async def test_edge_branches_abstain_never_guess():
    """Missing year columns, non-numeric cells, zero-denominator %-change, all-stopword row labels,
    and table-less hits ahead of the table — every edge abstains (fallback None) or skips cleanly."""
    lookup = _lookup()
    # A text hit BEFORE the table hit is skipped, then the table resolves.
    ctx = GenerationContext(_FakeRetrieval([_text_hit(), _table_hit()]), llm=None)
    out = await lookup.reason(Query(text="change in goodwill impairment between 2018 and 2019"), ctx)
    assert out.answer == "-12830"

    # Year not in the table (2017) -> unresolvable -> abstain.
    ctx = GenerationContext(_FakeRetrieval([_table_hit()]), llm=None)
    out = await lookup.reason(Query(text="change in goodwill impairment between 2017 and 2018"), ctx)
    assert out.abstained is True

    # Dash-valued cell + stopword-only row label -> unresolvable -> abstain.
    ctx = GenerationContext(_FakeRetrieval([_hit_with(_mixed_table())]), llm=None)
    out = await lookup.reason(Query(text="change in deferred revenue between 2018 and 2019"), ctx)
    assert out.abstained is True

    # Zero denominator for %-change -> unresolvable -> abstain.
    zero = Table(n_rows=2, n_cols=3, cells=[
        TableCell(id="h1", row=0, col=1, is_column_header=True, text="2019"),
        TableCell(id="h2", row=0, col=2, is_column_header=True, text="2018"),
        TableCell(id="r", row=1, col=0, is_row_header=True, text="Deferred revenue"),
        TableCell(id="a", row=1, col=1, text="5"),
        TableCell(id="b", row=1, col=2, text="0"),
    ])
    ctx = GenerationContext(_FakeRetrieval([_hit_with(zero)]), llm=None)
    out = await lookup.reason(Query(text="percentage change in deferred revenue between 2018 and 2019"), ctx)
    assert out.abstained is True


def _plan_table() -> Table:
    """Columns 2023 / 2022; rows "Plan A", "Plan B", "Revenue"."""
    cells = [
        TableCell(id="h0", row=0, col=0, is_column_header=True, text=""),
        TableCell(id="h1", row=0, col=1, is_column_header=True, text="2023"),
        TableCell(id="h2", row=0, col=2, is_column_header=True, text="2022"),
    ]
    for r, (label, a, b) in enumerate([("Plan A", "10", "8"), ("Plan B", "7", "5"), ("Revenue", "30", "20")], 1):
        cells += [
            TableCell(id=f"r{r}", row=r, col=0, is_row_header=True, text=label),
            TableCell(id=f"r{r}a", row=r, col=1, text=a),
            TableCell(id=f"r{r}b", row=r, col=2, text=b),
        ]
    return Table(n_rows=4, n_cols=3, cells=cells)


def test_an_article_does_not_resolve_a_single_letter_row():
    """MED-07: "a" in the question must not cover half of "Plan A" — an unknown row is unresolved (refused /
    delegated), never guessed."""
    reasoner = TableLookupReasoner(TableLookupReasoner.Config(fallback=None))
    assert reasoner._best_row(_plan_table(), "What is a change in operating costs from 2022 to 2023?") == (None, "")
    assert reasoner._best_row(_plan_table(), "What is a change in revenue from 2022 to 2023?") == (3, "Revenue")


def test_a_named_letter_still_picks_between_sibling_rows():
    reasoner = TableLookupReasoner(TableLookupReasoner.Config(fallback=None))
    assert reasoner._best_row(_plan_table(), "What was the change in Plan B from 2022 to 2023?") == (2, "Plan B")
    assert reasoner._best_row(_plan_table(), "What was the change in Plan A from 2022 to 2023?") == (1, "Plan A")
