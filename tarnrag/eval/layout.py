"""Layout-aware retrieval eval on TAT-QA — Option #2 (differentiators), PR-1.

TAT-QA pairs a financial **table** with associated **paragraphs**, and labels each question with where its
answer lives (``answer_from`` ∈ table / text / table-text). That label is ground truth for a question the
MOTHRAG benchmarks can't ask: **does retrieval surface the answer-bearing *element* — and does it find
*tables* as well as it finds text?** We build one shared corpus of every table + paragraph (so retrieval has
to pick the right element out of thousands, not ~6), then measure **source-hit@k** (did the top-k include the
gold answer-source element), segmented by ``answer_from``. Comparing dense vs hybrid (``--hybrid``) shows
whether BM25's exact-token matching helps *table* retrieval specifically (cell tokens embed weakly).

Extractive only: arithmetic / count answers have no retrievable source span, so they're filtered out.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from bausatz import ComponentFactory

from tarnrag.core.engine.config import GENERATION_PIPELINE, DatabaseSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmark_runner import _safe_answer
from tarnrag.eval.generation import GenEvalQuery, GenEvalReport, _aggregate, _score
from tarnrag.generation import GenerationContext, Reasoner
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.types import Query
from tarnrag.storage.repository import DocumentRepository

_EXTRACTIVE = frozenset({"span", "multi-span"})  # arithmetic/count have no retrievable source span

# Attribution measurement: read the answer with a single_hop reasoner, then have an LLM judge whether each
# cited span supports its claim (grounding). ``grounded_rate`` is the answer-level attribution precision.
def _attribution_pipeline(table_view: str) -> dict:
    """The attribution generation spec with the table view PINNED on both the reasoner and the judge —
    a measurement never inherits a default (the P2 default-flip lesson)."""
    return {
        "class_name": "generation_pipeline",
        "reasoner": {"class_name": "single_hop", "table_view": table_view},
        "grounding_checker": {"class_name": "llm_grounding", "table_view": table_view},
    }


@dataclass
class TatQaQuery:
    """One extractive TAT-QA question: the text, gold answer span(s), the gold answer-source element ids
    (table uid and/or paragraph uids), and the ``answer_from`` segment the source-hit is reported under."""

    question: str
    answers: list[str]
    gold_source_ids: list[str]
    answer_from: str  # "table" | "text" | "table-text"


def render_table(grid: list[list[str]]) -> str:
    """Linearize a TAT-QA table grid (rows of cells) to text — cells joined by ``|`` per row — preserving the
    exact cell tokens (so BM25 can match them) and the row structure. (The ``table_json`` extractor produces
    the same rendering as the element text, so the native path keeps the BM25 tokens identical.)"""
    return "\n".join(" | ".join(c for c in row) for row in grid)


def load_tatqa(
    rows: list[dict], *, limit: int | None = None
) -> tuple[list[tuple[str, str, str]], list[TatQaQuery]]:
    """Map TAT-QA records (``{table, paragraphs, questions}``) → a deduped corpus of
    ``(uid, content, kind)`` elements — each table as its **JSON grid** (``kind='table'``, ingested through
    the native ``table_json`` extractor so chunks carry a real ``Table``), each paragraph as text — and the
    extractive ``TatQaQuery`` list. ``limit`` caps records. Pure (no network) — pass ``rows`` from
    :func:`stream_tatqa` or a fixture."""
    corpus: dict[str, tuple[str, str]] = {}  # uid -> (content, kind)
    queries: list[TatQaQuery] = []
    for rec in rows[:limit] if limit else rows:
        table = rec["table"]
        tuid = table["uid"]
        corpus[tuid] = (json.dumps(table["table"]), "table")
        para_uid_by_order = {}
        for p in rec["paragraphs"]:
            corpus[p["uid"]] = (p["text"], "text")
            para_uid_by_order[str(p["order"])] = p["uid"]
        for q in rec["questions"]:
            if q.get("answer_type") not in _EXTRACTIVE:
                continue
            af = q.get("answer_from", "")
            gold: set[str] = set()
            if af in ("table", "table-text"):
                gold.add(tuid)
            if af in ("text", "table-text"):
                gold.update(para_uid_by_order[str(o)] for o in q.get("rel_paragraphs", []) if str(o) in para_uid_by_order)
            if not gold:
                continue
            queries.append(
                TatQaQuery(question=q["question"], answers=list(q.get("answer", [])), gold_source_ids=sorted(gold), answer_from=af)
            )
    return [(uid, content, kind) for uid, (content, kind) in corpus.items()], queries


def stream_tatqa(limit: int | None = None) -> list[dict]:
    """Stream TAT-QA validation records from HuggingFace (integration-only; the rest of the module is pure)."""
    from datasets import load_dataset

    split = f"validation[:{limit}]" if limit else "validation"
    return list(load_dataset("next-tat/TAT-QA", split=split))


@dataclass
class SourceHitReport:
    """source-hit@k overall and per ``answer_from`` segment (n + hit-rate each). ``hits`` carries the
    per-query outcomes (aligned with the query list), so callers can re-segment by their own labels
    (S1: the structural lexical/semantic slice) without re-running retrieval."""

    k: int
    n: int
    source_hit: float
    by_segment: dict[str, tuple[int, float]] = field(default_factory=dict)  # segment -> (n, hit-rate)
    hits: list[bool] = field(default_factory=list)  # per-query, aligned with the input query list


async def build_tatqa_index(
    corpus: list[tuple[str, str, str]], settings: Settings, *, db_path: str
) -> tuple[DocumentRepository, Embedder]:
    """Ingest each ``(uid, content, kind)`` element with ``source_id == uid`` (so ``document_id`` maps a
    retrieved chunk back to its element) into a persistent, **cached** store. Tables (``kind='table'``)
    go through the native ``table_json`` extractor — chunks carry a real ``Table`` (persisted cells,
    contextualizable at embed time); paragraphs stay on the text path. Mirrors ``build_corpus_index``:
    a full store is reused (skip the slow embed), a partial one is rebuilt."""
    settings.ID_POLICY = "caller"  # ingest each element under its uid, so document_id maps back for source-hit
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    repo = await DocumentRepository.create(
        DatabaseSettings(document_url=f"sqlite:///{db_path}"), settings.EMBEDDING_DIMENSION
    )
    await IngestionEngine.ensure_index_meta(repo, embedder)
    ingest = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
    if len(await ingest.list_documents()) == len(corpus):  # already indexed — reuse
        return repo, embedder
    for summary in await ingest.list_documents():
        await ingest.delete_document(summary.document_id)
    await ingest.ingest_content(
        [
            {
                "source_id": uid,
                "content": content,
                "title": uid,
                "source_type": kind,
                **({"extractor": "table_json"} if kind == "table" else {}),
            }
            for uid, content, kind in corpus
        ]
    )
    return repo, embedder


async def tatqa_source_hit(
    queries: list[TatQaQuery],
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    settings: Settings,
    k: int = 10,
    concurrency: int = 8,
) -> SourceHitReport:
    """For each query, retrieve top-``k`` over the shared corpus and score a **source-hit** = any retrieved
    chunk's ``document_id`` is a gold answer-source element. Reported overall + segmented by ``answer_from``.
    Bounded ``concurrency`` (independent read-only retrievals)."""
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    sem = asyncio.Semaphore(concurrency)

    async def _hit(q: TatQaQuery) -> bool:
        async with sem:
            results = await retrieval.search(Query(text=q.question, top_k=k))
        return any(r.document_id in set(q.gold_source_ids) for r in results)

    hits = await asyncio.gather(*(_hit(q) for q in queries))
    by_segment: dict[str, tuple[int, float]] = {}
    for seg in sorted({q.answer_from for q in queries}):
        seg_hits = [h for h, q in zip(hits, queries) if q.answer_from == seg]
        by_segment[seg] = (len(seg_hits), sum(seg_hits) / len(seg_hits)) if seg_hits else (0, 0.0)
    return SourceHitReport(
        k=k, n=len(queries), source_hit=(sum(hits) / len(hits)) if hits else 0.0,
        by_segment=by_segment, hits=list(hits),
    )


def format_source_hit(report: SourceHitReport, *, tag: str = "") -> str:
    """Render the source-hit report: overall + a row per answer_from segment (table / text / table-text)."""
    lines = [
        f"TAT-QA layout-aware retrieval{(' ' + tag) if tag else ''}  (source-hit@{report.k}, n={report.n})",
        f"{'segment':<14}{'n':>6}{'source-hit':>12}",
        "-" * 32,
    ]
    for seg, (n, hit) in report.by_segment.items():
        lines.append(f"{seg:<14}{n:>6}{hit:>12.3f}")
    lines.append(f"{'OVERALL':<14}{report.n:>6}{report.source_hit:>12.3f}")
    return "\n".join(lines)


def _to_eval_query(q: TatQaQuery) -> GenEvalQuery:
    """A TAT-QA query as a labeled ``GenEvalQuery`` — gold answer span(s) drive F1/EM (max over them), and
    ``query_type`` carries ``answer_from`` so attribution can be segmented by table vs text."""
    answers = list(q.answers)
    return GenEvalQuery(
        text=q.question,
        answer=answers[0] if answers else "",
        answer_aliases=answers[1:],
        answer_contains=answers,
        supporting=answers,  # citation coverage = is the gold answer span present in the cited evidence
        query_type=q.answer_from,
    )


async def tatqa_attribution(
    queries: list[TatQaQuery],
    llm: LanguageModel,
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    settings: Settings,
    table_view: str,
    concurrency: int = 8,
) -> tuple[GenEvalReport, dict[str, GenEvalReport]]:
    """Answer each question over the shared corpus and have an LLM judge the citations: returns the overall
    ``GenEvalReport`` (F1/EM + ``grounded_rate`` = attribution precision + ``citation_coverage``) and one per
    ``answer_from`` segment — so a table-vs-text attribution gap is visible. ``table_view`` is REQUIRED
    (pinned onto reasoner + judge — a measurement never inherits a default). Bounded ``concurrency``."""
    settings.components[GENERATION_PIPELINE] = _attribution_pipeline(table_view)
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    generation = GenerationEngine.assemble(retrieval, llm, settings)
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: TatQaQuery):
        eq = _to_eval_query(q)
        async with sem:
            result = await _safe_answer(generation.answer(Query(text=eq.text, purpose=eq.purpose)))
        return q.answer_from, _score(eq, result)

    scored = await asyncio.gather(*(_one(q) for q in queries))
    overall = _aggregate([s for _, s in scored])
    by_segment = {
        seg: _aggregate([s for sg, s in scored if sg == seg])
        for seg in sorted({q.answer_from for q in queries})
    }
    return overall, by_segment


def format_attribution(overall: GenEvalReport, by_segment: dict[str, GenEvalReport]) -> str:
    """Render attribution: F1 / EM / attrib (grounded_rate) / cite (citation_coverage), per segment + overall."""
    head = f"{'segment':<14}{'n':>5}{'F1':>8}{'EM':>8}{'attrib':>8}{'cite':>8}"

    def row(name: str, r: GenEvalReport) -> str:
        return f"{name:<14}{r.n:>5}{r.token_f1:>8.3f}{r.exact_match:>8.3f}{r.grounded_rate:>8.3f}{r.citation_coverage:>8.3f}"

    lines = ["TAT-QA attribution (single_hop + llm_grounding)", head, "-" * 51]
    lines += [row(seg, r) for seg, r in by_segment.items()]
    lines.append(row("OVERALL", overall))
    return "\n".join(lines)


# ---------------- the numeric slice (PP-6): arithmetic questions, answered without an LLM ----------


@dataclass
class TatQaNumericQuery:
    """One arithmetic TAT-QA question: the text, the numeric gold (already in cell units — TAT-QA's
    ``scale`` rides along for reporting), and the gold source segmentation."""

    question: str
    gold: float
    scale: str
    answer_from: str


def load_tatqa_numeric(rows: list[dict], *, limit: int | None = None) -> list[TatQaNumericQuery]:
    """The arithmetic questions (the slice ``load_tatqa`` filters out) with parseable numeric golds —
    the ready-made eval for the deterministic ``table_lookup`` reasoner."""
    queries: list[TatQaNumericQuery] = []
    for rec in rows[:limit] if limit else rows:
        for q in rec["questions"]:
            if q.get("answer_type") != "arithmetic":
                continue
            raw = q.get("answer")
            try:
                gold = float(str(raw).replace(",", ""))
            except (TypeError, ValueError):
                continue
            queries.append(
                TatQaNumericQuery(
                    question=q["question"], gold=gold, scale=q.get("scale", ""),
                    answer_from=q.get("answer_from", ""),
                )
            )
    return queries


@dataclass
class NumericReport:
    """The table_lookup eval: how many questions it ATTEMPTED (vs abstained — never guessed), and
    numeric exact-match both among attempts and overall (an abstention counts as a miss overall)."""

    n: int
    attempted: int
    correct: int

    @property
    def attempted_rate(self) -> float:
        return self.attempted / self.n if self.n else 0.0

    @property
    def em_attempted(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0

    @property
    def em_overall(self) -> float:
        return self.correct / self.n if self.n else 0.0


async def tatqa_numeric(
    queries: list[TatQaNumericQuery],
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    settings: Settings,
    reasoner_spec: dict,
    llm: LanguageModel | None = None,
    k: int = 10,
    concurrency: int = 8,
) -> NumericReport:
    """Run every arithmetic question through the caller-PINNED ``reasoner_spec`` (a measurement never
    inherits a default) — e.g. the LLM-free ``table_lookup`` posture, or a reader leg for comparison
    (pass its ``llm``). Numeric exact-match = equality after rounding both sides to 2 decimals (the
    gold precision); an abstention counts as unattempted."""
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    reasoner = ComponentFactory.get().create_as(reasoner_spec, Reasoner)
    ctx = GenerationContext(retrieval, llm=llm)
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: TatQaNumericQuery) -> tuple[bool, bool]:
        async with sem:
            # The reasoner runs directly (not through a GenerationPipeline), so the grounding flag the
            # pipeline would add is set here: a reader leg must not see grounding-disallowed chunks.
            out = await reasoner.reason(Query(text=q.question, top_k=k, grounds_generation=True), ctx)
        if out.abstained:
            return False, False
        try:
            got = float(out.answer.replace(",", "").replace("$", "").replace("%", "").strip())
        except (ValueError, AttributeError):
            return True, False
        return True, round(got, 2) == round(q.gold, 2)

    scored = await asyncio.gather(*(_one(q) for q in queries))
    return NumericReport(
        n=len(scored),
        attempted=sum(a for a, _ in scored),
        correct=sum(c for _, c in scored),
    )


def format_numeric(report: NumericReport, *, tag: str = "") -> str:
    return "\n".join([
        f"TAT-QA numeric (table_lookup, no LLM){(' ' + tag) if tag else ''}  (n={report.n})",
        f"attempted   {report.attempted:>5}  ({report.attempted_rate:.3f})",
        f"EM attempted{report.em_attempted:>12.3f}",
        f"EM overall  {report.em_overall:>12.3f}   (abstentions count as misses)",
    ])
