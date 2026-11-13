"""Smoke test for Part II · Example 07 — grounding check + abstain.

Two tests, matching the example's two halves:
  * the CONSTRUCTED abstain is deterministic (heuristic grounding + a static LLM, no store) — it runs
    everywhere, including CI, and is the core lesson: an unsupported claim is refused;
  * the LIVE verification grounds the good answer — gated on the embedder AND OPENAI_LLM_KEY (a real call).
"""

from __future__ import annotations

import os

import pytest

from examples.common import load_env
from examples.part_ii.example_07 import run

load_env()  # surface OPENAI_LLM_KEY from the repo-root .env for the live test's skipif


async def test_example_07_grounding_abstains_on_an_unsupported_claim():
    # Deterministic — no model, no key: a planted warranty claim disjoint from a maintenance passage.
    result = await run.planted_abstain()
    assert result.proof[0].grounded is False   # the grounding check flags the unsupported claim
    assert result.abstained is True            # and the abstain policy refuses rather than answer


@pytest.mark.requires_model
@pytest.mark.skipif(not os.environ.get("OPENAI_LLM_KEY"), reason="OPENAI_LLM_KEY not set (live LLM call)")
async def test_example_07_live_verification_grounds_the_good_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))
    good_verified, refused = await run.main()
    assert good_verified is True   # the answerable question is verified-grounded and still answered
    assert refused is True         # main() also runs the constructed abstain
