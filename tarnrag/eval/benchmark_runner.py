"""Run the MOTHRAG multi-hop QA benchmarks through tarn.rag's generation engine and score them.

Distractor protocol: for each question we ingest *only its own* candidate passages into an ephemeral store
(isolated, so retrieval can't leak across questions), run the configured ``GenerationPipeline``, and score
the answer with the generation harness's token-F1 / exact-match (the metric MOTHRAG reports). One embedder
and one store are reused across questions (the store is cleared each question); the LLM reader is injected
(an OpenAI-compatible Llama endpoint to match MOTHRAG, or a ``StaticLanguageModel`` for offline validation).

This is **not** a controlled MOTHRAG replication — different embedder/reader and the lean reasoners (no
ensemble / bridge / ChainFilter) — but a baseline of where tarn.rag's current stack stands. See
``MOTHRAG_PUBLISHED`` for the numbers to compare against.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager

from bausatz import ComponentFactory
from tarnrag.core.engine.config import DatabaseSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmarks import BenchItem
from tarnrag.eval.generation import (
    GenEvalReport,
    _aggregate,
    _score,
    format_generation_reports,
)
from tarnrag.generation import GenerationContext, GenerationPipeline, GenerationResult
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.types import Query
from tarnrag.storage.repository import DocumentRepository

logger = logging.getLogger(__name__)


async def _safe_answer(answer: Awaitable[GenerationResult]) -> GenerationResult:
    """Await an answer, returning an empty result (scored as a miss) instead of crashing the whole sweep on a
    single question's failure — a hard LLM error after retries, a context-length 400, etc. Failures are
    logged so a high count is visible (they shouldn't be silently absorbed)."""
    try:
        return await answer
    except Exception as e:
        logger.warning("benchmark answer failed; scoring as empty: %s: %s", type(e).__name__, e)
        return GenerationResult(answer="")


# MOTHRAG's published F1 / EM (paper Table 1 + the EM line), Llama-3.3-70B reader, n=1000/dataset.
MOTHRAG_PUBLISHED: dict[str, dict[str, float]] = {
    "hotpotqa": {"f1": 0.781, "em": 0.648},
    "2wiki": {"f1": 0.763, "em": 0.682},
    "musique": {"f1": 0.505, "em": 0.406},
}

# The Phase-2 bridge retrieval pipeline (vs the lean dense-only default): multi-query expansion + RRF, then
# an LLM relevance judge. Both call ``ctx.llm`` (the engine injects it from Settings). Set as the
# RETRIEVAL_PIPELINE spec to measure the bridge's effect on recall (hit) and downstream F1/EM.
BRIDGE_RETRIEVAL: dict[str, object] = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "multi_query"}],
    "fuser": {"class_name": "identity"},
    "reranker": {"class_name": "llm_judge"},
}

# Hybrid retrieval (vs the lean dense-only default): dense KNN + sparse BM25, RRF-fused. BM25 matches exact
# entity tokens — the multi-hop *bridge* entity that's lexically present but embeds weakly against the
# question — so it targets the dense-only recall ceiling directly, at no LLM cost.
HYBRID_RETRIEVAL: dict[str, object] = {
    "class_name": "retrieval_pipeline",
    "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
    "fuser": {"class_name": "rrf"},
}


@asynccontextmanager
async def _eval_engines(settings: Settings) -> AsyncIterator[tuple[IngestionEngine, RetrievalEngine]]:
    """One ephemeral SQLite store + one shared embedder + the ingestion/retrieval engines over it — the
    substrate every benchmark question is ingested into and retrieved from (the store is cleared per
    question by the caller). Everything is released on exit — the engines' own resources, the embedder,
    the store."""
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    with tempfile.TemporaryDirectory() as tmp:
        repo = await DocumentRepository.create(
            DatabaseSettings(document_url=f"sqlite:///{tmp}/eval.db"), settings.EMBEDDING_DIMENSION
        )
        ingest: IngestionEngine | None = None
        retrieval: RetrievalEngine | None = None
        try:
            await IngestionEngine.ensure_index_meta(repo, embedder)
            ingest = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
            retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
            yield ingest, retrieval
        finally:
            if retrieval is not None:
                await retrieval.aclose()  # its own LLM client; the injected store/embedder stay ours
            if ingest is not None:
                await ingest.aclose()
            embedder.close()
            await repo.disconnect()


async def _ingest_passages(ingest: IngestionEngine, item: BenchItem) -> None:
    """Replace the store's contents with this question's passages — distractor isolation: each question
    retrieves over only its own candidates."""
    for summary in await ingest.list_documents():
        await ingest.delete_document(summary.document_id)
    await ingest.ingest_content(
        [{"content": text, "title": title, "source_type": "text"} for title, text in item.passages]
    )


async def run_benchmark(
    items: list[BenchItem], llm: LanguageModel, *, settings: Settings | None = None
) -> GenEvalReport:
    """Answer each ``BenchItem`` over its own passages and return the aggregate generation report
    (token-F1 / EM / grounded-rate / citation-coverage). ``llm`` is the injected reader; ``settings`` drives
    the embedder + the ``GENERATION_PIPELINE`` spec (reasoner / grounding)."""
    settings = settings or Settings(_env_file=None)
    async with _eval_engines(settings) as (ingest, retrieval):
        generation = GenerationEngine.assemble(retrieval, llm, settings)
        per_query = []
        for item in items:
            await _ingest_passages(ingest, item)
            result = await _safe_answer(generation.answer(Query(text=item.query.text, purpose=item.query.purpose)))
            per_query.append(_score(item.query, result))
        return _aggregate(per_query)


def _reasoner_spec(name: str, grounding: str | None) -> dict[str, object]:
    """A ``generation_pipeline`` spec for reasoner ``name``; for ``grounded_retrieval``, ``grounding``
    overrides its grounding checker (e.g. ``llm_grounding`` for a sharper γ signal than the heuristic)."""
    reasoner: dict[str, object] = {"class_name": name}
    if grounding and name == "grounded_retrieval":
        reasoner["grounding_checker"] = {"class_name": grounding}
    return {"class_name": "generation_pipeline", "reasoner": reasoner}


async def sweep_benchmark(
    items: list[BenchItem],
    llm: LanguageModel,
    *,
    reasoners: list[str] | None = None,
    settings: Settings | None = None,
    grounding: str | None = None,
) -> dict[str, GenEvalReport]:
    """Run several reasoners over the *same* per-question passages and score each — the Phase-0 config
    comparison (``single_hop`` vs ``iterative`` vs ``decomposition``), reader + embedder held fixed and
    grounding off (it's a reasoner comparison). Each question is ingested once and answered by every
    reasoner over one shared ``GenerationContext``; returns one report per reasoner name."""
    settings = settings or Settings(_env_file=None)
    reasoners = reasoners or ["single_hop", "iterative", "decomposition"]
    specs = {r: _reasoner_spec(r, grounding) for r in reasoners}
    factory = ComponentFactory.get()
    async with _eval_engines(settings) as (ingest, retrieval):
        ctx = GenerationContext(retrieval, llm)
        pipelines = {name: factory.create_as(spec, GenerationPipeline) for name, spec in specs.items()}
        per_query: dict[str, list] = {name: [] for name in pipelines}
        for item in items:
            await _ingest_passages(ingest, item)
            query = Query(text=item.query.text, purpose=item.query.purpose)
            for name, pipeline in pipelines.items():
                result = await _safe_answer(pipeline.answer(query, ctx))
                per_query[name].append(_score(item.query, result))
        return {name: _aggregate(pq) for name, pq in per_query.items()}


# --------------------------------------------------------------------------- #
# Shared-corpus (fullwiki-style) harness: ingest one corpus, retrieve-only per question.
# --------------------------------------------------------------------------- #
async def build_corpus_index(
    corpus: list[tuple[str, str]], settings: Settings, *, db_path: str
) -> tuple[DocumentRepository, Embedder]:
    """Ingest the corpus once into a persistent store at ``db_path`` and return ``(repo, embedder)`` for the
    retrieve-only run. **Cached**: if the store already holds the whole corpus (matching doc count) the
    embedding is skipped and the existing index reused — so baseline-vs-bridge runs and reruns share one
    build. A partial / stale store is cleared and rebuilt. The caller disconnects the returned repo."""
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    repo = await DocumentRepository.create(
        DatabaseSettings(document_url=f"sqlite:///{db_path}"), settings.EMBEDDING_DIMENSION
    )
    await IngestionEngine.ensure_index_meta(repo, embedder)
    ingest = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
    try:
        existing = await ingest.list_documents()
        if len(existing) == len(corpus):  # already indexed — reuse, skip the (slow) embedding
            return repo, embedder
        for summary in existing:  # clear a partial / stale build, then ingest fresh
            await ingest.delete_document(summary.document_id)
        await ingest.ingest_content(
            [{"content": text, "title": title, "source_type": "text"} for title, text in corpus]
        )
        return repo, embedder
    finally:
        await ingest.aclose()  # its own resources only — the repo + embedder go back to the caller


async def run_over_corpus(
    items: list[BenchItem],
    llm: LanguageModel,
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    settings: Settings,
    concurrency: int = 8,
) -> GenEvalReport:
    """Answer each item by retrieving over the *shared* pre-built corpus index (no per-question ingest) —
    the fullwiki-style setting. Scoring is unchanged; retrieval can now miss the gold entirely. Questions
    run with bounded ``concurrency`` — they're independent + read-only over the shared corpus, so the
    (I/O-bound) LLM calls overlap; results are order-independent, so the aggregate is unchanged."""
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    generation = GenerationEngine.assemble(retrieval, llm, settings)
    sem = asyncio.Semaphore(concurrency)

    async def _one(item: BenchItem):
        async with sem:
            result = await _safe_answer(
                generation.answer(Query(text=item.query.text, purpose=item.query.purpose))
            )
        return _score(item.query, result)

    try:
        per_query = await asyncio.gather(*(_one(item) for item in items))  # gather preserves item order
    finally:
        await retrieval.aclose()  # its own LLM client; the injected store/embedder stay the caller's
    return _aggregate(list(per_query))


async def sweep_over_corpus(
    items: list[BenchItem],
    llm: LanguageModel,
    repo: DocumentRepository,
    embedder: Embedder,
    *,
    reasoners: list[str] | None = None,
    settings: Settings,
    concurrency: int = 8,
    grounding: str | None = None,
) -> dict[str, GenEvalReport]:
    """The reasoner sweep over the *shared* pre-built corpus (retrieve-only) — one report per reasoner.
    Each (item, reasoner) runs with bounded ``concurrency`` (independent + read-only over the shared corpus)."""
    reasoners = reasoners or ["single_hop", "iterative", "decomposition"]
    specs = {r: _reasoner_spec(r, grounding) for r in reasoners}
    factory = ComponentFactory.get()
    retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    ctx = GenerationContext(retrieval, llm)
    pipelines = {name: factory.create_as(spec, GenerationPipeline) for name, spec in specs.items()}
    sem = asyncio.Semaphore(concurrency)

    async def _one(name: str, pipeline: GenerationPipeline, item: BenchItem):
        async with sem:
            result = await _safe_answer(
                pipeline.answer(Query(text=item.query.text, purpose=item.query.purpose), ctx)
            )
        return name, _score(item.query, result)

    tasks = [_one(name, p, item) for item in items for name, p in pipelines.items()]
    per_query: dict[str, list] = {name: [] for name in pipelines}
    try:
        for name, scored in await asyncio.gather(*tasks):  # item-major order preserved per reasoner
            per_query[name].append(scored)
    finally:
        await retrieval.aclose()  # its own LLM client; the injected store/embedder stay the caller's
    return {name: _aggregate(pq) for name, pq in per_query.items()}


def format_comparison(reports: dict[str, GenEvalReport]) -> str:
    """A table of tarn.rag's F1 / EM per dataset against MOTHRAG's published numbers (+ deltas + averages).
    Keys are dataset names (``hotpotqa`` / ``2wiki`` / ``musique``)."""
    header = f"{'dataset':<12}{'n':>6}{'hit':>8}{'F1':>8}{'F1*':>8}{'ΔF1':>8}{'EM':>8}{'EM*':>8}{'ΔEM':>8}"
    lines = [header, "-" * len(header), "(* = MOTHRAG published, Llama-3.3-70B reader; hit = answer recall)"]
    hits, f1s, ems, mf1s, mems = [], [], [], [], []
    for name, r in reports.items():
        m = MOTHRAG_PUBLISHED.get(name, {})
        mf1, mem = m.get("f1", float("nan")), m.get("em", float("nan"))
        lines.append(
            f"{name:<12}{r.n:>6}{r.content_hit:>8.3f}{r.token_f1:>8.3f}{mf1:>8.3f}{r.token_f1 - mf1:>+8.3f}"
            f"{r.exact_match:>8.3f}{mem:>8.3f}{r.exact_match - mem:>+8.3f}"
        )
        hits.append(r.content_hit)
        f1s.append(r.token_f1)
        ems.append(r.exact_match)
        mf1s.append(mf1)
        mems.append(mem)
    if reports:
        lines.append(
            f"{'AVG':<12}{'':>6}{_avg(hits):>8.3f}{_avg(f1s):>8.3f}{_avg(mf1s):>8.3f}{_avg(f1s) - _avg(mf1s):>+8.3f}"
            f"{_avg(ems):>8.3f}{_avg(mems):>8.3f}{_avg(ems) - _avg(mems):>+8.3f}"
        )
    return "\n".join(lines)


def format_sweep(dataset: str, reports: dict[str, GenEvalReport]) -> str:
    """Render a reasoner sweep for one dataset: the per-reasoner metric table (one row each) + MOTHRAG's
    published F1/EM for that dataset as the reference to beat."""
    m = MOTHRAG_PUBLISHED.get(dataset, {})
    ref = (
        f"MOTHRAG ({dataset}, Llama-3.3-70B): "
        f"F1={m.get('f1', float('nan')):.3f}  EM={m.get('em', float('nan')):.3f}"
    )
    return f"{format_generation_reports(reports)}\n\n{ref}"


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
