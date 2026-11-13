"""Smoke test for Part II · Example 05 — query routing (Act A capstone).

Asserts the lesson: routing each query to the per-type-best method beats every fixed pipeline on the
labeled set, and the structural classifier reproduces most of the labels with none of its own. Gated on
the embedder (``requires_model``) — routing uses dense/sparse, so no reranker model is needed.
"""

from __future__ import annotations

import pytest

from examples.part_ii.example_05 import run

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_05_routing_beats_the_fixed_pipelines():
    reports, classifier_correct = await run.main()
    # Routing's ceiling is the per-type best, so it beats the fixed hybrid and is no worse than dense/sparse.
    assert reports["routed"].hit_at_k > reports["hybrid"].hit_at_k
    assert reports["routed"].hit_at_k >= reports["dense"].hit_at_k
    assert reports["routed"].hit_at_k >= reports["sparse"].hit_at_k
    # The structural classifier reproduces most labels with none of its own (7/8 on this set).
    assert classifier_correct >= 6
