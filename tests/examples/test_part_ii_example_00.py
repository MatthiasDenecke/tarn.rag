"""Smoke test for Part II · Example 00 — ingest the shared corpus.

Runs the example's ``main()`` against an isolated store (via ``EXAMPLES_DATA_DIR``, which
``common.load_config`` honours) and checks the corpus is fully indexed. Gated on ``requires_model``.
"""

from __future__ import annotations

import pytest

from examples.part_ii.example_00 import run

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Redirect the example's SQLite store into a temp dir, so the test never writes into the tree."""
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_00_ingests_corpus_into_the_base_store():
    docs = await run.main()
    assert len(docs) == 12  # every corpus-2 markdown file
    assert all(d.chunk_count >= 1 and d.embedding_count >= 1 for d in docs)
    assert sum(d.chunk_count for d in docs) > 12  # the recursive chunker split the longer docs
