"""DocumentStatusReader — the rollup that composes job_status with data facts."""

from tarnrag.contracts import DocumentFacts
from tarnrag.storage.status import DocumentStatusReader


class _Facts:
    def __init__(self, facts):
        self._facts = facts

    async def document_facts(self, document_id):
        return self._facts


class _Jobs:
    def __init__(self, rows):
        self._rows = rows

    async def document_jobs(self, document_id):
        return self._rows


def _reader(facts, jobs):
    return DocumentStatusReader(_Jobs(jobs), _Facts(facts))


async def test_unknown_document_is_none():
    assert await _reader(DocumentFacts(False, 0, 0), []).document_status("d") is None


async def test_pending_when_queued_but_no_data():
    s = await _reader(DocumentFacts(False, 0, 0), [{"status": "queued"}]).document_status("d")
    assert s["status"] == "pending"


async def test_in_progress_when_data_incomplete():
    s = await _reader(DocumentFacts(True, 3, 1), [{"status": "completed"}]).document_status("d")
    assert s["status"] == "in_progress"


async def test_complete_when_all_embedded():
    s = await _reader(DocumentFacts(True, 3, 3), [{"status": "completed"}]).document_status("d")
    assert s == {"document_id": "d", "status": "complete", "chunk_count": 3, "embedding_count": 3}


async def test_failed_overrides_counts():
    s = await _reader(DocumentFacts(True, 3, 3), [{"status": "failed"}]).document_status("d")
    assert s["status"] == "failed"


async def test_verbose_includes_jobs():
    rows = [{"status": "completed", "job_id": "j1"}]
    s = await _reader(DocumentFacts(True, 1, 1), rows).document_status("d", verbose=True)
    assert s["jobs"] == rows


async def test_complete_once_every_leaf_is_embedded():
    """Section parents get no vector, so completeness is measured against the leaves (``leaf_count``)."""
    s = await _reader(DocumentFacts(True, 5, 3, leaf_count=3), [{"status": "completed"}]).document_status("d")
    assert s["status"] == "complete" and s["chunk_count"] == 5  # all chunks still reported
