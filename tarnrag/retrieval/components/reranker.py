"""The Reranker component family: a second-pass re-scoring of the retrieved shortlist.

A ``Reranker`` is an optional, config-driven ``Component`` that re-scores and re-orders the surviving
results just before truncation. ``cross_encoder`` scores each (query, passage) pair with a cross-encoder
model — more accurate than the first-pass retrievers, but too costly to run over the whole index, so it
runs last (after filter + auto-merge, over the shortlist). The model is injected at call time via the
``RetrievalContext`` (``ctx.cross_encoder``), like the embedder — the component itself stays a pure
strategy. The §9 decision is to **keep** the first-pass component scores: the rerank score is added under
its own key, the originals stay visible.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Literal

from bausatz import Component

from tarnrag.contracts import RetrievalResult
from tarnrag.core.exceptions import RetrievalError
from tarnrag.core.parsing import extract_json
from tarnrag.core.resources.llm import Prompt
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class Reranker(Component):
    """Port: re-score + re-order the result shortlist (best first)."""

    class Config(Component.Config):
        """Base reranker config; concrete rerankers pin ``class_name``."""

    @abstractmethod
    async def rerank(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        """Return the results re-scored and re-ordered (best first)."""


class CrossEncoderReranker(Reranker):
    """Re-score each result with a cross-encoder (query × text), set it as the new ``score``, and
    re-order. The original first-pass scores stay in ``component_scores``; the rerank score is added
    under ``score_key``. The model comes from ``ctx.cross_encoder`` (configured in ``RerankSettings``)."""

    class Config(Reranker.Config):
        class_name: Literal["cross_encoder"] = "cross_encoder"
        score_key: str = "cross_encoder"  # key the rerank score is surfaced under in component_scores
        top_n: int = 20  # rerank only the top-N incoming candidates (parity with llm_judge): bounds the
        # CPU cost at corpus scale — an uncapped hybrid shortlist (~dense_k + sparse_k hydrated hits)
        # measured ~11s/query on CPU; the tail keeps its first-pass order below the reranked shortlist
        # and never reaches top_k (≪ top_n), so nothing is lost.

    config: CrossEncoderReranker.Config

    async def rerank(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        if not results:
            return results
        if ctx.cross_encoder is None:
            raise RetrievalError(
                "a cross_encoder reranker is configured but the RetrievalContext has no cross-encoder "
                "model — set the `rerank` settings (model dir) so the engine can build it"
            )
        shortlist, tail = results[: self.config.top_n], results[self.config.top_n :]
        # Cross-encoders are CPU-bound (ONNX) — score off the event loop, like query embedding.
        scores = await asyncio.to_thread(ctx.cross_encoder.score, query.text, [r.text for r in shortlist])
        rescored = [
            replace(r, score=s, component_scores={**r.component_scores, self.config.score_key: s})
            for r, s in zip(shortlist, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored + tail  # reranked shortlist first; the unranked tail trails (never reaches top_k)


_JUDGE_SYSTEM = (
    "Score how relevant each numbered passage is for answering the question, on a 0–10 scale (10 = directly "
    "contains the answer, 0 = irrelevant). Reply with a SINGLE JSON object and nothing else: "
    '{"scores": [{"passage": <number>, "score": <0-10>}, ...]}, covering every passage by its number.'
)


class LlmJudgeReranker(Reranker):
    """Bridge reranker: re-score the shortlist with an LLM relevance judge (query × passages, one batched
    call), set the judged score, and re-order (deterministic tie-break: score desc, then chunk_id asc).
    The LLM comes from ``ctx.llm`` — premium/economy tiers = swap the LLM spec. A passage the model doesn't
    score keeps 0. An LLM call per query (opt-in, off the lean default)."""

    class Config(Reranker.Config):
        class_name: Literal["llm_judge"] = "llm_judge"
        score_key: str = "llm_judge"  # key the judge score is surfaced under in component_scores
        top_n: int = 20  # judge only the top-N incoming candidates — bounds the prompt + cost at corpus
        # scale (multi-query over a large corpus can hand the reranker 100+ candidates), and the judge is
        # sharper over a shortlist. The tail keeps its first-pass order below the judged shortlist; it never
        # reaches ``top_k`` (≪ top_n), so nothing is lost.

    config: LlmJudgeReranker.Config

    async def rerank(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        if not results:
            return results
        if ctx.llm is None:
            raise RetrievalError(
                "an llm_judge reranker is configured but the RetrievalContext has no LLM — the engine "
                "injects it from `Settings.llm`; build the engine from Settings (or inject an llm)"
            )
        shortlist, tail = results[: self.config.top_n], results[self.config.top_n :]
        passages = "\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(shortlist))
        completion = await ctx.llm.complete(
            Prompt(system=_JUDGE_SYSTEM, user=f"Question: {query.text}\n\nPassages:\n{passages}")
        )
        scores = self._parse_scores(completion.text, len(shortlist))
        rescored = [
            replace(r, score=s, component_scores={**r.component_scores, self.config.score_key: s})
            for r, s in zip(shortlist, scores)
        ]
        rescored.sort(key=lambda r: (-r.score, r.chunk_id))
        return rescored + tail  # judged shortlist first; the unjudged tail trails (never reaches top_k)

    @staticmethod
    def _parse_scores(text: str, n: int) -> list[float]:
        """Map the reply's ``{"scores": [{"passage", "score"}, …]}`` to a per-result score list (0.0 for
        any passage the model omitted or scored out of range; bools rejected — they're ints in Python)."""
        out = [0.0] * n
        data = extract_json(text)
        rows: Any = data.get("scores") if isinstance(data, dict) else None
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            p, s = row.get("passage"), row.get("score")
            if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= n:
                continue
            if isinstance(s, bool) or not isinstance(s, int | float):
                continue
            out[p - 1] = float(s)
        return out
