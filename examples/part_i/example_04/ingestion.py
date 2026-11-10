"""Example 04 (ingestion) — ingest the corpus for the generation example (offline, SQLite).

The retrieval substrate for example 04's end-to-end generation. Ingestion is entirely offline (the local
ONNX embedder); only the generation step (``generation.py``) needs an LLM. The ingestion pipeline is read
from ``ingestion.json`` next to this script — same JSON-driven composition as example 02.

Run (from the repo root):

    python -m examples.part_i.example_04.ingestion
    python -m examples.part_i.example_04.generation
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from examples.common import base_settings, corpus, example_db, require_model
from tarnrag import DocumentStatus, TarnRag
from tarnrag.core.engine.config import INGESTION_PIPELINE

HERE = Path(__file__).resolve().parent
PIPELINE_SPEC = json.loads((HERE / "ingestion.json").read_text())
DOC_PATHS = sorted(corpus("corpus-1").glob("*.txt"))


async def main() -> list[DocumentStatus]:
    """Ingest corpus-1 with the JSON-configured pipeline; return each document's status."""
    require_model()
    db_path = example_db(__file__)
    settings = base_settings(db_path, components={INGESTION_PIPELINE: PIPELINE_SPEC})

    print(f"Ingesting {len(DOC_PATHS)} documents (pipeline from ingestion.json) into {db_path}\n")
    # `ingest` uses each file's stem as its document id (ID_POLICY="caller", the base_settings default).
    async with TarnRag(settings) as tarn:
        outcome = await tarn.ingest([str(p) for p in DOC_PATHS])

    statuses = outcome.value
    for issue in outcome.report.issues:
        print(f"  ! {issue.severity.value}: {issue.subject} — {issue.message}")
    for s in statuses:
        print(f"  {s.document_id:16}  {s.status:9}  chunks={s.chunk_count}  embeddings={s.embedding_count}")
    print("\nNext:  python -m examples.part_i.example_04.generation")
    return statuses


if __name__ == "__main__":
    asyncio.run(main())
