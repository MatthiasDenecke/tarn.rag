"""Part II · Example 04 — structure-aware chunking + auto-merge.

The first re-ingest: the Chunk stage switches from `recursive` to `structure_aware`, which builds a section
tree (leaf chunks + section parents). Because the representation changes, this builds its OWN store
(structured.db). Two things follow:

  * FIX — Example 03's fragmentation goes away: a section-spanning question retrieves the whole coherent
    *section* (the parent chunk), not a single fragment, so the full answer comes back;
  * AUTO-MERGE — when several sibling leaf fragments of one section are retrieved, they're collapsed into
    their parent, so the top-k isn't dominated by one document's pieces (more diverse results).

Run from the repo root::

    python -m examples.part_ii.example_04.run

Needs the embedder + the reranker model (see Example 03's header for the reranker fetch).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.common import corpus, load_config, require_model
from examples.part_ii._runner import Runner
from tarnrag import TarnRag

CONFIG = Path(__file__).resolve().parent / "config.yaml"


async def main() -> tuple[bool | None, int]:
    """Return (the fragmentation fix verdict, the number of distinct documents auto-merge freed room for)."""
    require_model()
    async with TarnRag(load_config(CONFIG)) as tarn:
        runner = Runner(tarn)
        runner.banner(
            "Example 04 · structure-aware chunking + auto-merge (re-ingest)",
            shows="structure-aware chunking builds a section tree, so a section-spanning question retrieves "
            "the whole coherent section — Example 03's fragment becomes a complete answer",
            fails=None,
            fixed_next="Example 05 — query routing: classify the query and send it to the per-type-best method",
        )
        await runner.ensure_corpus(corpus("corpus-2"))
        complete = await runner.show_retrieval(
            "what are all the steps in the reciprocating compressor startup procedure?",
            gold="loading sequence", label="fixed · the whole section, not a fragment", top_k=1,
        )
        # Auto-merge: a query that pulls several sibling fragments of one section — they consolidate into
        # the parent (watch the `merged` stage in the breakdown), freeing top-k slots for other documents.
        await runner.show_retrieval(
            "compressor startup and shutdown",
            label="auto-merge · sibling fragments consolidated into the section parent", top_k=6,
        )
        results = (await tarn.retrieve("compressor startup and shutdown", top_k=6)).value
        distinct = len({r.document_id for r in results})
        runner.note(f"{distinct} distinct documents in the top 6 — auto-merge freed the slots")
        return complete, distinct


if __name__ == "__main__":
    asyncio.run(main())
