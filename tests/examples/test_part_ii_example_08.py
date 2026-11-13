"""Smoke test for Part II · Example 08 — multi-hop reasoning (decomposition).

Asserts the lesson: a question whose answer spans two documents (TX-200 → GMV frame → ISO VG 68) is
answered by ``decomposition`` and NOT by ``single_hop`` — the bridge passage is crowded out of the top-k
by four near-identical frame specs, so one retrieve→read pass can never see it. Makes live LLM calls, so
it is gated on the embedder (``requires_model``) AND OPENAI_LLM_KEY — skipped in CI, run locally where the
repo-root .env provides the key.
"""

from __future__ import annotations

import os

import pytest

from examples.common import load_env
from examples.part_ii.example_08 import run

load_env()  # surface OPENAI_LLM_KEY from the repo-root .env for the skipif below

pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(not os.environ.get("OPENAI_LLM_KEY"), reason="OPENAI_LLM_KEY not set (live LLM call)"),
]


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLES_DATA_DIR", str(tmp_path))


async def test_example_08_multi_hop_reaches_the_bridge_single_hop_cannot():
    multi_hop_hit, single_hop_hit = await run.main()
    assert multi_hop_hit is True    # decomposition bridges the two docs → "ISO VG 68"
    assert single_hop_hit is False  # one retrieve→read pass never sees the GMV spec → "not specified"
