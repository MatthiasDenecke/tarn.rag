"""Smoke test for Part II · Example 03 — cross-encoder reranking.

Asserts the lesson: rerank recovers the paraphrase hybrid demoted (HIT) and keeps the part-number (HIT),
but a section-spanning question returns only a fragment (MISS) — which Example 04 (auto-merge) fixes.
Gated on the embedder (``requires_model``) AND the separate reranker model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.part_ii.example_03 import run

_RERANKER = Path("models/ms-marco-MiniLM-L-6-v2/model.onnx")
pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(
        not _RERANKER.exists(),
        reason="reranker model not fetched (python scripts/fetch_model.py "
        "--repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2)",
    ),
]


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_03_rerank_recovers_paraphrase_keeps_id_but_fragments():
    recovered, kept, fragmented = await run.main()
    assert recovered is True   # rerank promotes the paraphrased doc back into the top-k (hybrid demoted it)
    assert kept is True        # the opaque part number still hits
    assert fragmented is False  # the single best chunk is only a fragment — Example 04 (auto-merge) fixes it
