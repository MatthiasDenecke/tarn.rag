"""Example 01 — Ingestion (embedded mode, SQLite).

The smallest end-to-end ingestion you can run: read the shared sample corpus from
``examples/docs/``, configure tarnrag for *embedded* mode over a local SQLite file, and report
what was stored. Config and shared helpers live in ``examples/common.py``.

In embedded mode the whole pipeline (load -> clean -> chunk -> enrich -> embed -> persist) runs
in THIS process — no Postgres, no job queue — so each ingest call returns only once the
documents are fully chunked, embedded, and written to the SQLite store.

Setup (once, from the repo root):

    pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
    python scripts/fetch_model.py  # download the embedding model + tokenizer (needs network)

Run (from the repo root):

    python -m examples.part_i.example_01.ingestion

Then query the same store with the retrieval example (which reuses the same shared config).
"""

from __future__ import annotations

import asyncio

from examples.common import base_settings, corpus, example_db, require_model
from tarnrag import DocumentStatus, TarnRag

# The shared corpus this example ingests (examples/docs/corpus-1/*.txt).
CORPUS = corpus("corpus-1")
DOC_PATHS = sorted(CORPUS.glob("*.txt"))


async def main() -> list[DocumentStatus]:
    """Ingest the corpus and return each document's terminal status (also printed)."""
    require_model()
    db_path = example_db(__file__)  # next to this script (or under $EXAMPLES_DATA_DIR)
    settings = base_settings(db_path)

    print(f"Ingesting {len(DOC_PATHS)} documents from {CORPUS} into {db_path}\n")

    # `TarnRag` is the facade over the three engines; `async with` builds them (the embedded SQLite
    # store + in-process worker) and disposes the connection pool on exit. `ingest` takes file/dir
    # paths and, under ID_POLICY="caller" (the base_settings default), uses each file's stem as its
    # stable document id (e.g. "tank-inspection") — so re-running upserts in place, idempotent per id.
    async with TarnRag(settings) as tarn:
        # Embedded mode completes the whole pipeline before returning, so each status is terminal.
        outcome = await tarn.ingest([str(path) for path in DOC_PATHS])

    statuses = outcome.value
    for issue in outcome.report.issues:  # e.g. a path that didn't resolve, or a doc that failed
        print(f"  ! {issue.severity.value}: {issue.subject} — {issue.message}")
    for status in statuses:
        print(
            f"  {status.document_id:16}  {status.status:9}  "
            f"chunks={status.chunk_count}  embeddings={status.embedding_count}"
        )
    print(f"\nStored {len(statuses)} documents. Now query the same store:")
    print("  python -m examples.part_i.example_01.retrieval")
    return statuses


if __name__ == "__main__":
    asyncio.run(main())
