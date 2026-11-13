"""Smoke test for Part II · Example 02 — hybrid retrieval.

Asserts the two Example-01 probes swap under hybrid: the opaque part number now HITS (sparse's exact
match) while the paraphrase now MISSES (sparse noise demotes the semantic answer). Model-gated.
"""

from __future__ import annotations

import pytest

from examples.part_ii.example_02 import run

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_02_hybrid_fixes_the_id_but_regresses_the_paraphrase():
    fixed, regressed = await run.main()
    assert fixed is True       # the opaque part number XQ-9920-A now hits (Example 01 missed it)
    assert regressed is False  # the paraphrase dense nailed now misses — Example 03 (rerank) recovers it
