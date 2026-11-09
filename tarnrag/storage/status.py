"""Document status read model — composes the job_status projection with the data facts.

Public document status (`pending | in_progress | complete | failed`) needs two *different*
things: the operational **job_status** projection (did anything fail / is it known?) and the
persisted **data facts** (how many chunks/embeddings exist). Both now live on the repository (one
store), but the read model still depends on **two narrow ports** — a ``JobStatusSource`` and a
``DocumentFactsSource`` (in ``tarnrag.contracts``) — not a fat "document storage" object (ISP —
each writer/caller depends only on what it uses, and the facts source can still be pointed
elsewhere). Only this thin read model composes the two.
"""

from __future__ import annotations

from typing import Any

from tarnrag.contracts import DocumentFacts, DocumentFactsSource, JobStatusSource


class DocumentStatusReader:
    """
    Derives public status by composing a ``JobStatusSource`` with a ``DocumentFactsSource``.
    """

    def __init__(self, jobs: JobStatusSource, facts: DocumentFactsSource):
        self._jobs = jobs
        self._facts = facts

    async def document_status(
        self, document_id: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        facts = await self._facts.document_facts(document_id)
        job_rows = await self._jobs.document_jobs(document_id)
        if not facts.present and not job_rows:
            return None  # unknown document
        states = {r["status"] for r in job_rows}
        out: dict[str, Any] = {
            "document_id": document_id,
            "status": self._rollup(facts, states),
            "chunk_count": facts.chunk_count,
            "embedding_count": facts.embedding_count,
        }
        if verbose:
            out["jobs"] = job_rows  # debug-only per-job breakdown
        return out

    @staticmethod
    def _rollup(facts: DocumentFacts, states: set[str]) -> str:
        if "failed" in states:
            return "failed"
        if not facts.present:
            return "pending"  # queued, no data persisted yet
        if states & {"queued", "processing"}:
            return "in_progress"  # a (re-)ingest is still running — the data on disk may be the old one
        # Complete once every retrieval unit has its vector (section parents are never embedded).
        indexable = facts.chunk_count if facts.leaf_count is None else facts.leaf_count
        if indexable and facts.embedding_count >= indexable:
            return "complete"
        return "in_progress"
