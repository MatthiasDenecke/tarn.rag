# tarn.rag

**tarnrag** — composable, DAG-based RAG **ingestion + retrieval + generation** engines: documents in, a
queryable vector index out, optionally a grounded answer with a proof tree. This is a console
application runnable in python from the command line, no other libraries are needed. The 'Quickstart' section
shows you how to get started and the configs in the example folder demonstrate how to configure more
complex systems that can be run from the command line.

- **Ingestion** — a config-driven pipeline (extract → enrich → clean → chunk → embed) writing one store
  (a §8 index: documents · chunks · vectors · FTS · metadata), embedded (SQLite + sqlite-vec/FTS5) or
  distributed (Postgres + pgvector, pgQueuer workers).
- **Retrieval** — config-driven, swappable Components: dense / sparse retrievers → RRF fusion → license &
  scope pre-filter → auto-merge → cross-encoder rerank, with layout-grade provenance on every hit.
- **Generation** — multi-hop reasoning (single-hop / iterative / decomposition) → grounding check →
  proof tree + evidence, over the retrieval substrate. LLM-pluggable; retrieval stays LLM-free by default.

## Install

```bash
pip install tarn-rag
```

The base install is the lightweight SQLite/embedded path. Backends and features are **extras**:

| Extra | Adds |
|---|---|
| `onnx` | local ONNX embedding (`onnxruntime` + `tokenizers`) |
| `embeddings-api` | hosted embedding APIs (OpenAI / Voyage / Gemini) |
| `parsers` | PDF/HTML document loaders (`pypdf`, `pdfplumber`, `beautifulsoup4`) |
| `docling` | high-fidelity layout-aware PDF extraction (heavy) |
| `generation` | the Anthropic LLM backend for the generation layer (`anthropic`) |
| `openai` | any OpenAI-compatible `/chat/completions` LLM backend (`httpx`) |
| `console` | the interactive `tarnrag` console UI (`rich`) |
| `benchmarks` | the HuggingFace loaders for the QA benchmarks (`datasets`) |
| `postgres` / `queue` | the distributed backends (`asyncpg` + `pgvector` / `pgqueuer`) |
| `all` | the common embedded runtime (`onnx`, `parsers`, `embeddings-api`, `generation`, `openai`, `console`) |

```bash
pip install "tarn-rag[onnx,parsers]"     # local embedder + PDF/HTML extraction
pip install "tarn-rag[all]"              # the full single-process runtime
pip install "tarn-rag[postgres,queue]"  # the distributed backends
```

## Quickstart — the interactive console

The fastest way to try tarn.rag end to end (needs the `console` extra, included in `all`). Start it with
one JSON config — a `Settings` document naming the database, embedder, LLM, and pipeline specs; a sample
lives at [`examples/console.config.json`](./examples/console.config.json):

```bash
tarnrag examples/console.config.json     # or: python -m tarnrag.console <config.json>
```

The LLM key is read from the environment (`ANTHROPIC_API_KEY`), not the config, and the sample config
expects the local embedding model fetched once with `scripts/fetch_model.py`. Then work the corpus at
the `tarn>` prompt:

```
tarn> ingest examples/docs/corpus-1/    # ingest (or re-ingest) files; a directory ingests the files in it
tarn> docs                  # list the ingested documents (id, chunks, embeddings)
tarn> retrieve <query>      # retrieval only — the ranked passages
tarn> explain <query>       # retrieval with its inner workings — per-retriever candidates, per-stage ranking
tarn> ask <query>           # retrieval + generation — the grounded answer + its proof tree
tarn> help                  # the full list (also: status, delete, quit)
```

Full walkthrough: [docs/tutorials/console-session.md](./docs/tutorials/console-session.md).

## Quickstart — the Python API

The high-level facade (`TarnRag`) wires ingestion + retrieval + generation over one store; each call returns
an `Outcome` (the value) plus a `Report` of any non-fatal issues:

```python
import asyncio
from tarnrag import TarnRag

async def main():
    async with TarnRag("config.json") as tarn:   # a JSON Settings file
        await tarn.ingest(["path/to/your/files"]) # ingest files / directories
        hits = await tarn.retrieve("how do I inspect a tank?", top_k=8)
        answer = await tarn.ask("…")             # grounded answer + proof tree (needs an LLM key)
        print(answer.value.answer)

asyncio.run(main())
```

Or drive the engines directly (each built from `Settings` via `create()`):

```python
from tarnrag import IngestionEngine, RetrievalEngine

engine = await IngestionEngine.create()
ids = await engine.ingest_paths(["/data/spec.pdf"])

async with await RetrievalEngine.create() as r:   # validates schema + embedding fingerprint
    results = await r.search_text("how do I inspect a tank?", top_k=8)
```

Full walkthrough: [docs/tutorials/getting-started.md](./docs/tutorials/getting-started.md) — and the
[documentation index](./docs/index.md) for everything beyond the quickstarts.

## The quality profile

The default retrieval pipeline is hybrid (dense + BM25, RRF-fused). For quality-sensitive
deployments, add the local cross-encoder reranker (measured: table source-hit 0.853 → 0.926 on
TAT-QA; costs ~seconds/query of CPU — see `doc/phases.md`):

```yaml
components:
  retrieval_pipeline:
    class_name: retrieval_pipeline
    retrievers: [{class_name: dense}, {class_name: sparse}]
    fuser: {class_name: rrf}
    reranker: {class_name: cross_encoder, top_n: 20}
```

(Fetch the reranker model once with `scripts/fetch_model.py`; it loads lazily on first use.)

**If the reranker's CPU cost is out of budget**, two measured cheaper levers exist (S1 sweep,
`doc/phases.md`) — on TAT-QA each recovers about half the reranker's lift (+0.015 overall,
+0.031–0.042 on tables over plain hybrid):

- **Sparse-weighted hybrid** (LLM-free): up-weight the BM25 arm where the corpus rewards
  exact-token matching — `fuser: {class_name: rrf, weights: {sparse: 3.0}}`. Only the ratios
  matter; tune the weight per corpus with `scripts/run_layout_eval.py --s1`.
- **Query-routed retrieval** (~1 LLM call per semantic query): a `routing_retrieval_pipeline`
  with the structural classifier — multi-query expansion helps natural-language questions but
  *hurts* keyword/exact-match ones, and routing applies each arm only where it wins:

  ```yaml
  components:
    retrieval_pipeline:
      class_name: routing_retrieval_pipeline
      classifier: {class_name: structural}
      routes:
        semantic:
          class_name: retrieval_pipeline
          retrievers: [{class_name: multi_query}, {class_name: sparse}]
          fuser: {class_name: rrf}
      default:
        class_name: retrieval_pipeline
        retrievers: [{class_name: dense}, {class_name: sparse}]
        fuser: {class_name: rrf}
  ```

  The route taken is recorded on the search trace (`explain`), so every answer can show *why* it
  was retrieved the way it was. Don't stack these under the reranker: measured, the CE subsumes
  both (identical scores with and without routing beneath it).

## Modes

- **`MODE='embedded'`** (default) — runs the whole pipeline in-process over SQLite; each ingest call
  processes to completion. Zero infrastructure.
- **`MODE='distributed'`** — enqueues to pgQueuer over Postgres; run `python run_worker.py` as one or more
  separate consumer processes.

Configuration is environment-driven (pydantic-settings, `GROUP__FIELD` convention, e.g. `EMBEDDING__MODEL`,
`DATABASE__DOCUMENT_URL`); see `.env.example`.

## Docs

Design specs and architecture notes live in [`doc/`](./doc): `FUNCTIONAL_REQUIREMENTS.md` (ingestion),
`ModusQ_RetrievalSubsystemSpec.md` + `retrieval-architecture-design.md` (retrieval), and
`generation-architecture-design.md` (generation). `CLAUDE.md` is the architecture-oriented contributor guide.

## License

MIT — see [LICENSE](./LICENSE).
