"""Example 02 — A pipeline configured from JSON (embedded mode, SQLite).

Same engine as example 01, but the ingestion *pipeline* is read from a JSON file (``pipeline.json``,
next to this script) instead of the built-in default. Edit that file — change a stage's parameters,
add or drop a stage, reorder them — and re-run; no Python changes.

How it works: a pipeline is just data — ``{"class_name": "pipeline", "stages": [...]}``. Dropping
that spec into ``Settings.components[INGESTION_PIPELINE]`` makes the engine build *that* pipeline.
Here ``pipeline.json`` uses small chunks (``chunk_size`` 80 vs the default 512), so the same short
corpus is split into MORE chunks than example 01 produced (one per doc).

One thing the JSON deliberately does NOT set: the Embed stage's embedding identity. That stays in
``Settings`` (shared with retrieval), so the engine injects it at build time and a hand-edited
pipeline can't drift from the retrieval-side embedder. Structure in JSON; embedding identity in
Settings.

Run (from the repo root, after the one-time setup in example 01):

    python -m examples.part_i.example_02.ingestion
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from examples.common import base_settings, corpus, example_db, require_model
from tarnrag import DocumentStatus, TarnRag
from tarnrag.core.engine.config import INGESTION_PIPELINE

# The ingestion pipeline as data — loaded from the JSON file next to this script.
PIPELINE_SPEC = json.loads((Path(__file__).resolve().parent / "pipeline.json").read_text())

CORPUS = corpus("corpus-1")
DOC_PATHS = sorted(CORPUS.glob("*.txt"))


async def main() -> list[DocumentStatus]:
    """Ingest the corpus with the JSON-configured pipeline; return each document's status."""
    require_model()
    db_path = example_db(__file__)
    # The one interesting line: the spec from pipeline.json drives the pipeline's composition + params.
    settings = base_settings(db_path, components={INGESTION_PIPELINE: PIPELINE_SPEC})

    print(f"Ingesting {len(DOC_PATHS)} documents with the pipeline from pipeline.json into {db_path}\n")

    # `ingest` uses each file's stem as its document id (ID_POLICY="caller", the base_settings default).
    async with TarnRag(settings) as tarn:
        outcome = await tarn.ingest([str(path) for path in DOC_PATHS])

    statuses = outcome.value
    for issue in outcome.report.issues:
        print(f"  ! {issue.severity.value}: {issue.subject} — {issue.message}")
    for status in statuses:
        print(
            f"  {status.document_id:16}  {status.status:9}  "
            f"chunks={status.chunk_count}  embeddings={status.embedding_count}"
        )
    total_chunks = sum(s.chunk_count for s in statuses)
    print(f"\nStored {len(statuses)} documents in {total_chunks} chunks (small-chunk pipeline; "
          "example 01's default made one per doc).")
    print("Query the same store:  python -m examples.part_i.example_02.retrieval")
    return statuses


if __name__ == "__main__":
    asyncio.run(main())
