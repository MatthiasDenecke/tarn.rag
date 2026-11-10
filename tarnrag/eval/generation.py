"""The generation eval harness — score generation-pipeline specs on a labeled question set.

The analog of the retrieval harness: ``sweep_generation`` runs many ``GenerationPipeline`` specs (different
reasoner / grounding configs — single-hop vs iterative vs decomposition, with / without a cascade) over the
same ``GenerationContext`` (the retrieval port + the LLM) and scores each on four axes:

- **answer quality** — content-hit (a gold phrase appears in the answer) + token-F1 + exact-match (SQuAD
  style), measured over the *answerable* queries;
- **grounded rate** — fraction of non-abstained answers the grounding check marked grounded;
- **abstention accuracy** — did it abstain iff the query is labeled unanswerable;
- **citation coverage** — fraction of a query's gold supporting phrases that appear in the answer's *cited*
  passages' text (the evidence quality, not just the answer).

Scoring is content-based (gold phrases), so labels survive re-ingestion. Running it for real needs an LLM;
the harness's own tests drive a canned ``StaticLanguageModel`` (deterministic, no key).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bausatz import ComponentFactory

from tarnrag.generation import GenerationContext, GenerationPipeline, GenerationResult
from tarnrag.retrieval.types import Purpose, Query


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class GenEvalQuery:
    """One labeled question. ``answer_contains`` (gold phrases) drives content-hit; ``answer`` (a gold
    string) drives token-F1 / exact-match — scored as the **max over ``answer`` + ``answer_aliases``** (the
    SQuAD/MuSiQue protocol, where several gold strings are acceptable); ``should_abstain`` marks a question
    unanswerable from the corpus; ``supporting`` (gold phrases) is what the answer's cited evidence covers."""

    text: str
    answer_contains: list[str] = field(default_factory=list)
    answer: str = ""
    answer_aliases: list[str] = field(default_factory=list)
    should_abstain: bool = False
    supporting: list[str] = field(default_factory=list)
    query_type: str = ""
    purpose: Purpose = Purpose.EXECUTION


@dataclass
class GenEvalSet:
    """A labeled question set — the ground truth a generation sweep is scored against."""

    queries: list[GenEvalQuery] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: list[dict]) -> GenEvalSet:
        """Build from plain dicts (only ``text`` is required; the rest default)."""
        return cls(
            [
                GenEvalQuery(
                    text=r["text"],
                    answer_contains=list(r.get("answer_contains", [])),
                    answer=r.get("answer", ""),
                    answer_aliases=list(r.get("answer_aliases", [])),
                    should_abstain=bool(r.get("should_abstain", False)),
                    supporting=list(r.get("supporting", [])),
                    query_type=r.get("query_type", ""),
                    purpose=Purpose(r.get("purpose", Purpose.EXECUTION)),
                )
                for r in records
            ]
        )

    @classmethod
    def from_json(cls, path: str | Path) -> GenEvalSet:
        return cls.from_records(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Metrics (pure functions over an answer / a result)
# --------------------------------------------------------------------------- #
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation + articles, split on whitespace."""
    return [t for t in _PUNCT.sub(" ", text.lower()).split() if t not in _ARTICLES]


def content_hit(answer: str, phrases: list[str]) -> bool:
    """Whether ``answer`` contains one of the gold ``phrases`` (case-insensitive)."""
    low = answer.lower()
    return any(p.lower() in low for p in phrases)


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between the predicted and gold answers (SQuAD-style)."""
    pred, ref = _normalize(prediction), _normalize(gold)
    if not pred or not ref:
        return float(pred == ref)  # both empty ⇒ 1.0; exactly one empty ⇒ 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized exact match between the predicted and gold answers."""
    return _normalize(prediction) == _normalize(gold)


def citation_coverage(result: GenerationResult, supporting: list[str]) -> float:
    """Fraction of the gold ``supporting`` phrases that appear in the answer's *cited* passages' text —
    the cited passages being those of ``result.evidence`` whose ``chunk_id`` the proof tree cites."""
    cited_ids = {c.chunk_id for step in result.proof for c in step.citations}
    cited_text = " ".join(e.text for e in result.evidence if e.chunk_id in cited_ids).lower()
    return sum(1 for p in supporting if p.lower() in cited_text) / len(supporting)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class GenQueryReport:
    """Per-query outcome. ``None`` ⇒ the metric isn't applicable here (no gold label, or — for the answer
    metrics — the query is labeled unanswerable)."""

    text: str
    abstained: bool
    abstention_correct: bool
    grounded: bool
    content_hit: bool | None
    token_f1: float | None
    exact_match: bool | None
    citation_coverage: float | None
    query_type: str = ""


@dataclass
class GenEvalReport:
    """A pipeline's aggregate generation metrics (means over the applicable queries) + the breakdown."""

    n: int
    content_hit: float
    token_f1: float
    exact_match: float
    grounded_rate: float
    abstention_accuracy: float
    citation_coverage: float
    per_query: list[GenQueryReport] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(q: GenEvalQuery, result: GenerationResult) -> GenQueryReport:
    """Score one answered query. The answer metrics are only applicable to *answerable* queries (a wrong
    abstention then scores 0, since the refusal won't contain the gold phrases)."""
    answerable = not q.should_abstain
    golds = [q.answer, *q.answer_aliases]  # several gold strings may be acceptable (SQuAD/MuSiQue: take the max)
    return GenQueryReport(
        text=q.text,
        abstained=result.abstained,
        abstention_correct=result.abstained == q.should_abstain,
        grounded=result.grounded,
        content_hit=content_hit(result.answer, q.answer_contains) if answerable and q.answer_contains else None,
        token_f1=max(token_f1(result.answer, g) for g in golds) if answerable and q.answer else None,
        exact_match=any(exact_match(result.answer, g) for g in golds) if answerable and q.answer else None,
        citation_coverage=citation_coverage(result, q.supporting) if q.supporting else None,
        query_type=q.query_type,
    )


def _aggregate(per_query: list[GenQueryReport]) -> GenEvalReport:
    """Mean each metric over the queries it applies to (``None`` values excluded)."""
    return GenEvalReport(
        n=len(per_query),
        content_hit=_mean([r.content_hit for r in per_query if r.content_hit is not None]),
        token_f1=_mean([r.token_f1 for r in per_query if r.token_f1 is not None]),
        exact_match=_mean([r.exact_match for r in per_query if r.exact_match is not None]),
        grounded_rate=_mean([r.grounded for r in per_query if not r.abstained]),
        abstention_accuracy=_mean([r.abstention_correct for r in per_query]),
        citation_coverage=_mean([r.citation_coverage for r in per_query if r.citation_coverage is not None]),
        per_query=per_query,
    )


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
async def evaluate_generation(
    pipeline: GenerationPipeline, ctx: GenerationContext, evalset: GenEvalSet
) -> GenEvalReport:
    """Answer every question through ``pipeline`` over ``ctx`` and score the four metric families."""
    per_query: list[GenQueryReport] = []
    for q in evalset.queries:
        result = await pipeline.answer(Query(text=q.text, purpose=q.purpose), ctx)
        per_query.append(_score(q, result))
    return _aggregate(per_query)


async def sweep_generation(
    specs: dict[str, dict[str, Any]], ctx: GenerationContext, evalset: GenEvalSet
) -> dict[str, GenEvalReport]:
    """Evaluate many named ``GENERATION_PIPELINE`` specs against the same context — the config comparison
    (reasoner / grounding variants over one retrieval port + LLM)."""
    factory = ComponentFactory.get()
    return {
        name: await evaluate_generation(factory.create_as(spec, GenerationPipeline), ctx, evalset)
        for name, spec in specs.items()
    }


def format_generation_reports(reports: dict[str, GenEvalReport]) -> str:
    """Render a sweep's reports as a comparison table — one row per generation-pipeline spec."""
    cols = [("hit", "content_hit"), ("f1", "token_f1"), ("em", "exact_match"),
            ("grnd", "grounded_rate"), ("abst", "abstention_accuracy"), ("cite", "citation_coverage")]
    header = f"{'pipeline':<22}" + "".join(f"{label:>8}" for label, _ in cols)
    lines = [header, "-" * len(header)]
    for name, r in reports.items():
        lines.append(f"{name:<22}" + "".join(f"{getattr(r, attr):>8.3f}" for _, attr in cols))
    if reports:
        lines.append(f"\n(n={next(iter(reports.values())).n} queries)")
    return "\n".join(lines)
