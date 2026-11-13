"""Smoke test for Part II · Example 04 — structure-aware chunking + auto-merge.

Asserts the lesson: the section-spanning question now returns the whole section (the fragment from
Example 03 is complete), and auto-merge consolidates sibling fragments so the top-k stays diverse.
Gated on the embedder (``requires_model``) AND the reranker model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.part_ii.example_04 import run

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


async def test_example_04_structure_aware_completes_section_and_auto_merge_diversifies():
    complete, distinct = await run.main()
    assert complete is True   # the whole startup section comes back (Example 03 returned only a fragment)
    assert distinct >= 5      # auto-merge consolidated the redundant section fragments → diverse top-k
