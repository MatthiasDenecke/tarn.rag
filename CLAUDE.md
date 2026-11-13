# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**tarnrag** — a composable, DAG-based RAG **ingestion + retrieval + generation** library: documents in, a
queryable vector index out, optionally a grounded answer with a proof tree. There is no HTTP layer here; a
consuming REST API lives in the separate **tiqtasq.backend** repo. The public surface is three engines plus
a high-level facade (`TarnRag`) that wires all three over one store:

```python
from tarnrag import TarnRag                                  # the high-level facade (ingest/retrieve/ask)
from tarnrag import IngestionEngine, RetrievalEngine, run_worker, Query, DocumentStatus
```

Data flow:

```
IngestionEngine.create() → ingest_paths/content/streams → PipelineOrchestrator (walks the DAG,
  creates jobs) → queue (InMemory in embedded mode · pgQueuer in distributed) → IngestionWorker(s)
  → stages (load+extract → enrich → clean → chunk → embed) → DocumentRepository (§8 index + job_status, one store)

RetrievalEngine.create() → await search → retrievers (dense/sparse, license-filtered) → fuse → hydrate
  → auto-merge → rerank → top_k → ranked, provenance-bearing results

GenerationEngine.create() → await answer → reason (retrieve↔read) → ground-check → proof tree + evidence
```

Design specs: `doc/FUNCTIONAL_REQUIREMENTS.md` (ingestion), `doc/ModusQ_RetrievalSubsystemSpec.md` +
`doc/retrieval-architecture-design.md` (retrieval), and `doc/generation-architecture-design.md` (generation).
Retrieval methods, generation steps, extractors, chunkers, and enrichers are all config-driven **Components**
(the standalone [`bausatz`](https://pypi.org/project/bausatz/) package, `>=0.3` — `Component` + `ComponentFactory` + `Registry`, imported directly) composed by spec under `Settings.components`.
Container children are **typed Configs** (`list[Retriever.Config]`, `Chunker.Config`, …): a spec tree validates in one
pass, and `_build_children` turns each child Config into a component with `factory.build_as(cfg, Family)`; a
directly-constructed container calls `self.ensure_built()` before first use.

## Using the engines

Both facades are built from `Settings` — no hand-wiring:

```python
from tarnrag import IngestionEngine, RetrievalEngine

# Ingestion (MODE='embedded' default → runs the whole pipeline in-process):
engine = await IngestionEngine.create()              # or .create(settings)
ids = await engine.ingest_paths(["/data/spec.pdf"])  # -> list[str]; also ingest_content / ingest_streams
# Identity vs content are separate. ID_POLICY pins the (stable) document id: 'uuid' (engine-
# assigned, default) or 'caller' (you pass source_ids=[...]). Content dedup is its own query:
#   if await engine.find_by_content_hash(engine.content_hash_of_file(path)):
#       ...                                          # these exact bytes are already ingested
st = await engine.status(ids[0])                     # -> DocumentStatus
docs = await engine.list_documents()                 # -> list[DocumentSummary] (id, content_hash, counts)
await engine.delete_document(ids[0])                 # remove a doc + its chunks/embeddings/jobs
await engine.aclose()                                # or: async with await IngestionEngine.create() as engine: ...

# Retrieval (async; reads the §8 index ingestion built):
async with await RetrievalEngine.create() as r:      # validates schema + embedding fingerprint
    hits = await r.search_text("how do I inspect a tank?", top_k=8)
```

- **`create()` is the entry point** for both engines (reads `Settings`). Bare constructors and
  `RetrievalEngine.open(...)` are low-level seams for tests / custom wiring.
- **`MODE='embedded'`** (default) runs in-process (InMemory queue) — each ingest call processes to
  completion. **`MODE='distributed'`** enqueues to pgQueuer; run `python run_worker.py` as one or
  more separate consumer processes.
- **Both engines are async:** ingestion and retrieval both run on the async repository (async
  SQLAlchemy; pgQueuer in distributed mode). `RetrievalEngine.search` / `search_text` are `async`
  (the query embed is thread-offloaded since ONNX is CPU-bound and releases the GIL); lifecycle
  follows the async idiom (`aclose` + `async with`). The portable SQLite file stays C++-consumable —
  a future sync C++ reader opens it directly, independent of the Python engine's async-ness.
- **Jobs are internal** — never in the public surface. The per-job breakdown is available only via
  the debug-gated `IngestionEngine.document_jobs` (raises unless `APP__DEBUG`).

## Package layout

```
tarnrag/
├── core/         # infra: engine/ (config,
│                 #   Engine base, observability), resources/ (Embedder · CrossEncoder · LanguageModel),
│                 #   exceptions, hashing
├── contracts/    # cross-boundary shared kernel: dtos · ports · results · structure · index_meta
├── storage/      # status.py (DocumentStatusReader) · repository/ (base · postgres · sqlite · chunk_provenance)
├── ingestion/    # components/ (extraction · chunking · enrichment), pipeline/ (stage bases · clean · embed),
│                 #   engine/ (engine+run_worker · worker · orchestrator · queue · batch · result_sink · jobs · types)
├── retrieval/    # components/ (retriever · fuser · merger · reranker · classifier · license_policy),
│                 #   pipeline/ (searcher · pipeline · router), engine/ (engine · protocol), types
├── generation/   # components/ (reasoner · grounding · assembler), pipeline/, engine/, context · types
├── eval/         # eval harnesses: retrieval (metrics · dataset · harness), generation (generation),
│                 #   public benchmarks (benchmarks · benchmark_runner), layout/TAT-QA (layout)
├── tarnrag.py    # TarnRag — high-level facade + composition root over the three engines
├── report.py · console.py · _cli.py  # Outcome/Report/Issue/Severity · rich console · entry wrapper
run_worker.py            # distributed consumer entry: asyncio.run(run_worker())
scripts/fetch_model.py   # fetch the ONNX model + tokenizer into the model dir
scripts/run_benchmarks.py · run_layout_eval.py   # QA-benchmark + TAT-QA layout eval CLIs
```

`tarnrag/__init__.py` re-exports the public API; consumers import from `tarnrag`, not the submodules.

## Architecture — the decoupled seams (preserve these)

- **Distributed execution, one job per `(item, stage)`.** Fan-out stages enqueue one downstream job
  per output item (chunker → *m* chunks → *m* embed jobs). In-flight items travel **inline in the
  job payload** — no temp item store.
- **Queue ports.** Two role-segregated ABCs: **`JobEnqueuer`** (`enqueue`, the producer, used by the
  orchestrator) and **`JobConsumer`** (`set_handler` / `run`, the consumer runtime). The worker is a
  registered handler, not a puller. **`PgQueuerJobQueue`** is the only file that imports pgQueuer
  (which owns SKIP LOCKED / retries / NOTIFY / dead-lettering); **`InMemoryJobQueue`** is the
  production embedded-mode queue (and the test double — no Postgres). The consumer hands the worker
  a **`Batch`** (`ingestion/engine/jobs.py`) — homogeneous (all jobs share one `stage_name`, enforced
  by the constructor); the InMemory queue groups each wave into real multi-job batches, the pgQueuer
  adapter still dispatches single-job batches. Keep the port a delegating seam — never reimplement
  queue mechanics in it.
- **Three-layer split:**
  - **Worker = compute only** (`ingestion/engine/worker.py`). A pure handler holding only the coordinator
    (no queue, no run loop): `handle_batch(batch)` gets the stage via
    `BatchCoordinator.get_stage(...)` (the DAG's **long-lived** instance, so `EmbedStage`'s model
    loads once), runs it, and reports to a **`BatchContext`** from `begin_batch(batch)`:
    `ctx.submit()` results, then `ctx.complete()` (or `ctx.fail()` + re-raise on compute error).
    Depends **only** on the two batch ABCs — never the orchestrator/repo/queue/`ResultSink`.
    Raising → the queue requeues (recovery).
  - **`ResultSink` = persistence** (`ingestion/engine/result_sink.py`). Output-side; batches and writes
    results. Composed inside the `BatchContext` — the worker never touches it.
    `async finalize() -> FinalizationOutcome`.
  - **Orchestrator = `BatchCoordinator` + lifecycle + DAG walking** (`ingestion/engine/orchestrator.py`).
    `begin_batch` records `processing`; `ctx.complete()` finalizes the sink, records status, and
    enqueues downstream jobs — or records `failed` and **raises** so the queue requeues. Also owns
    `ingest_documents`. Downstream is enqueued only after upstream persists (implicit dependencies).
- **Stages stay pure** — no DB/queue access (the worker observes them). All stages subclass
  `PipelineStage` (`ingestion/pipeline/pipeline.py`); typed bases `MapperStage` (1→1, `map()`) /
  `FilterStage` (1→{0,1}, `should_keep()`) merge metadata automatically (subclasses return *updates*,
  never mutate the bag). Fan-out stages are **container stages** that build Component children:
  `LoadAndParse` (extractors), `Enrich` (enrichers), `Chunk` (a `Chunker`, default `structure_aware`).
- **Layout-aware extraction (Component seam).** `LoadAndParseStage` (`ingestion/components/extraction/`)
  routes a `source_kind` to an `Extractor` Component that produces a `StructuredDocument` (ordered
  elements + geometry + header paths + tables): `plain_text` · `markdown` · `html` (BeautifulSoup) ·
  `pdf_text` (pdfplumber, fast tier) · `docling` (high-fidelity, opt-in `[docling]` extra). Routes are
  config (`Config.routes`); a per-document `metadata['extractor']` override picks one inline. The
  structure-aware chunker reads `item.document` and emits chunks carrying `ChunkProvenance` (header path,
  geometry, the auto-merge `parent_chunk_id`).
- **The shared embedder** (`core/resources/embedder.py`). `Embedder` ABC + `OnnxEmbedder` (prefix →
  tokenize → ONNX CPU → pool(mask) → L2; lazy `onnxruntime`/`tokenizers`) + HTTP API backends
  (`embedder_api.py`: OpenAI/Voyage/Gemini), selected by `settings.embedding.provider` via
  `Embedder.create`. The **same** embedder embeds passages (ingestion) and queries (retrieval),
  guaranteeing pipeline identity; `config_fingerprint()` is recorded in `index_meta` and retrieval refuses
  to `open()` on mismatch. Model configurable via `settings.embedding` (default `gte-small`); fetch
  artifacts with `scripts/fetch_model.py`. `CrossEncoder` (reranker) and `LanguageModel` (generation) are
  sibling `Resource`s in `core/resources/`.
- **The §8 index + status read model** (`storage/`). The retrieval index lives in the
  `DocumentRepository` itself (one store) — embedded: a single SQLite file (`index_meta`, `documents`,
  `chunks`, `vec_chunks` via sqlite-vec, `fts_chunks` via FTS5, `method_chunks`); distributed:
  Postgres with dense retrieval on `embeddings` (pgvector). `ChunkStore` is the persistence ABC the
  repository implements (where the ingestion sinks write). `DocumentStatusReader` (`storage/status.py`)
  composes two **narrow ports** — `JobStatusSource` (the repo's `job_status`) + `DocumentFactsSource`
  (presence + counts) — and owns the rollup (ISP: only this read model sees both), now both backed by
  the repository.
- **Database agnosticism** (`storage/repository/`). All document storage goes through
  `DocumentRepository` (SQLAlchemy 2.0 Core, async): shared tables + dialect-agnostic CRUD in
  `base.py` (read-model assembly lives in its companion `read_assembly.py`, held as `repo.reads`);
  `postgres.py` / `sqlite.py` supply only the dialect hooks — `_driver_url`, `_vector_type`,
  `_encode_vector` / `_decode_vector`, `_dialect_insert` (the `ON CONFLICT` upsert construct),
  `_before_create_schema` / `_after_create_schema`,
  `_index_chunk_text`, `_clear_chunk_index`, `_count_doc_embeddings`,
  `_embedding_counts_by_document` — plus the search half of the `RetrievalStore` port (`dense_knn` /
  `sparse_search`; `hydrate` is shared Core in the base). Postgres uses pgvector; SQLite uses sqlite-vec
  (`vec_chunks`) for dense KNN + FTS5 for sparse. Both sparse builders render one parse
  (`core.text.lexical_query`: required phrases AND an OR of the remaining terms), so the dialects agree
  on what a query means. Selection is driven by `settings.database.document_url` (a `postgres` substring
  → Postgres, else SQLite). Put shared logic in the base, dialect specifics in the hooks.
- **Transactional guarantees & idempotency.** `store_document_with_chunks` / `store_chunks` /
  `store_embeddings` are atomic. Documents are keyed by `metadata['source_id']` (UNIQUE, **stable** —
  set per `ID_POLICY` = `uuid` | `caller`); re-ingesting upserts the document and replaces its
  chunks/embeddings (delete chunks → cascade removes embeddings) — re-runs never duplicate.
  `store_chunks` likewise replaces a document's chunks (a retried Chunk job can't add a second set), chunk
  ids are content-addressed (`uuid5(document, ordinal, content hash)`), and every upsert is one atomic
  `INSERT … ON CONFLICT DO UPDATE` (no update-then-insert race between concurrent workers). Each
  document also stores a `content_hash` column (sha256 of submitted content); `find_by_content_hash`
  queries it for content dedup, independent of identity. `list_documents` (inventory + counts) and
  `delete_document` round out the document admin surface; `delete_document_jobs` clears the
  job-status rows (the engine's `delete_document()` does both).
- **Two databases.** `settings.database.queue_url` (pgQueuer) is separate from
  `settings.database.document_url` (document/chunk/embedding storage). Never conflate them.
- **Observability is optional.** Core logic must work with `observability=None` (guard every `self.obs`
  call); `Observability.create(settings.observability)` returns the configured adapter or `None` when
  disabled. Adapters: `structured_logging` (JSON-lines over stdlib logging — one machine-parseable line
  per metric/log event); any other `type` (incl. the not-yet-implemented `prometheus`) installs the no-op.
- **Retrieval is config-driven Components.** `RetrievalEngine.search` delegates to a `Searcher` built from
  `Settings.components[RETRIEVAL_PIPELINE]` — a `RetrievalPipeline` (parallel `Retriever`s {dense/sparse} →
  `Fuser` {identity/rrf, `(score desc, chunk_id asc)` tie-break} → hydrate → optional `Merger` {auto-merge}
  → optional `Reranker` {cross-encoder} → `top_k`), or a `RoutingRetrievalPipeline` (a `QueryClassifier`
  dispatches per `query_type`). License/scope filtering is a **pre-filter inside the retrievers**:
  `Query.permitted_filter()` (via the configured `LicensePolicy`) builds a `ChunkFilter` that
  `dense_knn`/`sparse_search` apply in SQL, over-fetching to backfill past dropped chunks. Comparing
  methods = swapping the `RETRIEVAL_PIPELINE` spec.

## Conventions

- **Config** (`core/engine/config.py`). `Settings` (pydantic-settings) nests per-component sub-models —
  `settings.embedding`, `settings.rerank`, `settings.llm`, `settings.database`, `settings.worker`,
  `settings.observability` — read from env via the `GROUP__FIELD` convention (e.g. `EMBEDDING__MODEL`,
  `DATABASE__DOCUMENT_URL`). Cross-cutting `MODE`, `EMBEDDING_DIMENSION`, `UPLOAD_DIR`, `ID_POLICY`
  stay top-level/flat. `Settings.components` holds the pipeline specs (`ingestion_pipeline` /
  `retrieval_pipeline` / `generation_pipeline` / `license_policy`), default-filled by a `model_validator`.
  Another `model_validator` pins the backend: `distributed` requires Postgres + `DATABASE__QUEUE_URL`;
  `embedded` requires SQLite. See `.env.example`. Each resource/repo is built via a `create()` from its
  config slice (`Embedder.create`, `DocumentRepository.create`).
- **Type annotations (Python 3.12):** builtin generics (`list`, `dict`, `tuple`, `type`) and
  `X | None` — never `typing.List` / `Dict` / `Optional`. Import `Iterator` / `Iterable` / `Callable`
  from `collections.abc`; keep `Any` / `Literal` from `typing`. Use `datetime.now(UTC)`.
- **Interfaces:** prefer **ABCs** (`abc.ABC` + `@abstractmethod`) over `typing.Protocol`;
  implementations inherit explicitly.

## Documentation & examples (two trees, two jobs)

- **`docs/` is prose, organized by [Diátaxis](https://diataxis.fr/)** — `tutorials/` (learning) ·
  `how-to/` (a task) · `reference/` (lookup) · `explanation/` (why). Diátaxis organizes by *user
  need*, so **never add a folder named after an artifact type** (`docs/examples/`, `docs/guides/`) —
  it belongs to no quadrant. `docs/index.md` is the entry point; every new page must be linked from
  its section index.
- **`examples/` is runnable code** — scripts, configs, corpora. It is an *importable package*
  (`python -m examples.part_i.example_01.ingestion`; `tests/examples/` imports `examples.*`), which
  is why its dirs use **underscores** (`part_i`, `example_04`) while doc dirs use **hyphens**
  (`part-i` — not a valid Python identifier). **Don't move example code under `docs/`.** The docs
  link *to* the examples; they are companion material, not a docs section.
- **Prose goes in exactly one place.** A walkthrough of an example is a *tutorial* →
  `docs/tutorials/part-i/` or `part-ii/`. Each example dir keeps only a short **card** README: its
  config delta, the run commands, and a link to its tutorial page. Never explain the same idea in
  both — they drift.
- **Verify every number you publish.** Ranks, scores, chunk counts, and eval tables in docs must come
  from *running the example against the current code*, never copied from an older README. Library
  defaults move (the default retrieval pipeline changed dense-only → hybrid mid-2026, which silently
  invalidated every documented score). Run it, then write it.
- **Docs must not contradict measured findings.** `doc/phases.md` records the sweeps (e.g. S1: the
  cross-encoder subsumes routing, so routing under a reranker is pure cost). Check a claim against it
  before teaching the opposite.
- **Part II configs are fully explicit YAML — no reliance on library defaults.** Every stage and
  component parameter is enumerated, so a default change cannot move the results. One knob per rung;
  the **diff between two configs is the lesson**. Ingest once (rungs are config swaps over one store);
  only a *representation* change (chunker/enricher/extractor) gets its own store, and says so.
- **Adding an example:** the code dir (with `__init__.py`), a smoke test in `tests/examples/`, a
  tutorial page under `docs/tutorials/part-*/`, a link in that series' `index.md`, and a card README.

## Stack

Python 3.12 · **Pydantic v2** / pydantic-settings (`model_config = ConfigDict(...)` /
`SettingsConfigDict(...)`) · **SQLAlchemy 2.0 Core (async)** (asyncpg + `pgvector` for Postgres,
aiosqlite + numpy for SQLite) · **pgQueuer** (distributed queue) · **sqlite-vec** + FTS5 (the §8
index) · **ONNX** (`onnxruntime` / `tokenizers`) for embeddings · pytest. Heavy backends
(Postgres/pgQueuer, ONNX) import lazily so the SQLite/embedded path stays light — the Postgres repo
and pgQueuer adapter are imported only inside the relevant `create()`.

## Commands

Dev runs in the conda env **`tarn.rag`** (Python 3.12). Each Bash call is a fresh non-login shell, so
prefix with `conda run -n tarn.rag`:

```bash
conda run -n tarn.rag python -m pytest -q                                  # full suite
conda run -n tarn.rag python -m pytest tests/ingestion -q                  # a subset
conda run -n tarn.rag ruff check tarnrag tests                             # lint
conda run -n tarn.rag python -m py_compile $(find tarnrag -name "*.py")    # compile-check
```

The suite runs entirely on SQLite + InMemory queue (no Postgres/pgQueuer needed); the real ONNX
embedder test is gated on the model dir existing.
