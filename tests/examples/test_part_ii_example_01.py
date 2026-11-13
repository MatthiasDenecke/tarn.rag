"""Smoke test for Part II · Example 01 — dense-only retrieval.

Runs the example's ``main()`` against an isolated store and checks the failure is *real*: dense finds the
paraphrased doc (by meaning) but misses the opaque part number (which Example 02 repairs). Model-gated.
"""

from __future__ import annotations

import pytest

from examples.part_ii.example_01 import run

pytestmark = pytest.mark.requires_model


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_01_dense_hits_paraphrase_misses_opaque_id():
    good, bad = await run.main()
    assert good is True   # dense finds the paraphrased doc by meaning (a keyword search would miss it)
    assert bad is False   # dense misses the bare opaque part number — Example 02 (sparse + RRF) fixes it
