"""Smoke test for Part II · Example 06 — minimal generation.

Asserts the lesson: an answerable question comes back grounded in the gold fact, while the minimal
pipeline does NOT abstain on an unanswerable one (the limitation Example 07 addresses). Makes a live LLM
call, so it is gated on the embedder (``requires_model``) AND the OPENAI_LLM_KEY being set — it is skipped
in CI, run locally where the repo-root .env provides the key.
"""

from __future__ import annotations

import os

import pytest

from examples.common import load_env
from examples.part_ii.example_06 import run

load_env()  # surface OPENAI_LLM_KEY from the repo-root .env for the skipif below

pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(not os.environ.get("OPENAI_LLM_KEY"), reason="OPENAI_LLM_KEY not set (live LLM call)"),
]


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_06_grounded_answer_but_no_abstain():
    grounded_good, abstained_unanswerable = await run.main()
    assert grounded_good is True          # the answerable question is answered, grounded in "mechanical seal"
    assert abstained_unanswerable is False  # the minimal pipeline never abstains — Example 07 adds that
