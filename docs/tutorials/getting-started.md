# Getting started

This tutorial takes you from an empty environment to a grounded answer over your own documents,
using the Python API. Everything runs in-process over a local SQLite file — no services to stand up.

## 1. Install

tarn.rag needs Python 3.12+. The `all` extra bundles the common single-process runtime (local ONNX
embedder, PDF/HTML parsers, the Anthropic + OpenAI-compatible LLM backends, the console):

```bash
pip install "tarn-rag[all]"
```

## 2. Fetch the embedding model

The default embedder is a local ONNX model (`thenlper/gte-small`), so queries and passages embed
offline and identically. Fetch the model artifacts once:

```bash
python scripts/fetch_model.py
```

This places `model.onnx` + `tokenizer.json` under `./models/gte-small` (the default
`embedding.model_dir`).

## 3. Write a config

A config file is a `Settings` document — JSON or YAML, selected by extension. A minimal embedded
setup:

```json
{
  "MODE": "embedded",
  "ID_POLICY": "caller",
  "EMBEDDING_DIMENSION": 384,
  "embedding": {"model_dir": "./models/gte-small"},
  "database": {"document_url": "sqlite:///./my_index.db"}
}
```

`"ID_POLICY": "caller"` makes each file's name (without the extension) its document id, so ingesting
the same file again updates that document instead of adding a copy. Leave it out and every ingest
gets a fresh random id — the right choice when you track documents by the ids `ingest` returns.

Everything you don't specify falls back to a sensible default — including the ingestion pipeline
(extract → enrich → clean → chunk → embed), hybrid retrieval (dense + BM25, RRF-fused), and the
decomposition-based generation pipeline. See the
[settings reference](../reference/settings.md) for every knob.

## 4. Ingest, retrieve, ask

The high-level facade (`TarnRag`) wires ingestion + retrieval + generation over one store. Each
call returns an `Outcome`: the value it produced plus a `Report` of any non-fatal issues (an
unreadable file, for example, is reported — never silently skipped).

```python
import asyncio
from tarnrag import TarnRag

async def main():
    async with TarnRag("config.json") as tarn:
        ingested = await tarn.ingest(["path/to/your/files"])   # files and/or directories
        for issue in ingested.report.issues:
            print("note:", issue)

        hits = await tarn.retrieve("how do I inspect a tank?", top_k=8)
        for hit in hits.value:
            print(hit)

        answer = await tarn.ask("how do I inspect a tank?")    # needs an LLM key — see below
        print(answer.value.answer)

asyncio.run(main())
```

`ask` calls the generation layer, which needs an LLM. The default provider is Anthropic; export
`ANTHROPIC_API_KEY` before running (only the *name* of a key variable is ever configured — the key
itself stays out of config files). `ingest` and `retrieve` never need an LLM.

## 5. Where things landed

Everything lives in the single SQLite file you configured (`my_index.db`): documents, chunks,
dense vectors (sqlite-vec), the BM25 index (FTS5), and ingestion job status. With
`ID_POLICY: caller` (as configured above), re-ingesting a file replaces its document's chunks and
embeddings in place — re-runs never duplicate.

## Next steps

- Prefer a REPL? Do the same round trip in the [console](console-session.md).
- Tune retrieval quality: [swap the retrieval pipeline](../how-to/swap-the-retrieval-pipeline.md).
- Outgrow one process: [run distributed mode](../how-to/run-distributed-mode.md).
- Drive the engines directly (`IngestionEngine` / `RetrievalEngine` / `GenerationEngine`) when you
  need finer control — see the [architecture overview](../explanation/architecture.md).
