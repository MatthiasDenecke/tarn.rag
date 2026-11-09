# Settings & environment variables

`Settings` (pydantic-settings) is the single source of runtime configuration. Grouped fields read
from the environment via the `GROUP__FIELD` convention (`EMBEDDING__MODEL`,
`DATABASE__DOCUMENT_URL`); the few cross-cutting knobs stay flat. Source of truth:
[`tarnrag/core/engine/config.py`](../../tarnrag/core/engine/config.py).

## How configuration is loaded

- **From the environment** — `Settings()` (via `get_settings()`) reads OS env vars plus an optional
  `.env` file. Unrecognized variables are ignored, not errors.
- **From a file** — `Settings.from_file(path)` loads a JSON or YAML document (selected by
  extension: `.json` / `.yaml` / `.yml`). The file is authoritative: the ambient `.env` is ignored
  so the same config reproduces everywhere; OS env vars still supplement what the file omits
  (typically API keys). `TarnRag("config.json")` and the console use this path.

## Top-level (flat) fields

| Env var | Default | Description |
|---|---|---|
| `MODE` | `embedded` | `embedded` runs the whole pipeline in-process (InMemory queue); `distributed` enqueues to pgQueuer. |
| `EMBEDDING_DIMENSION` | `384` | The embedder's output dimension — the index and repository must match it. |
| `UPLOAD_DIR` | `./uploads` | Staging dir for streamed bytes (a shared volume in distributed mode). Embedded mode removes each staged copy once its ingest has run; distributed workers read them later, so expiring them there is the deployment's job. |
| `ID_POLICY` | `uuid` | How document ids are assigned: `uuid` (engine-assigned) or `caller` (you supply every `source_id`). Strict — a mismatch fails ingestion. |

## `app` — process flags

| Env var | Default | Description |
|---|---|---|
| `APP__DEBUG` | `false` | Gates the debug-only surface (e.g. `IngestionEngine.document_jobs`). |

## `embedding` — the shared embedder

One embedder embeds both passages (ingestion) and queries (retrieval); its identity feeds the index
fingerprint (see [hosted embeddings](../how-to/use-hosted-embeddings.md)).

| Env var | Default | Description |
|---|---|---|
| `EMBEDDING__PROVIDER` | `onnx` | `onnx` (local, offline) · `openai` / `voyage` / `gemini` (HTTP APIs) · `hash` (deterministic, model-free — CI/demos only). |
| `EMBEDDING__MODEL` | `thenlper/gte-small` | HF id (onnx) or API model name. |
| `EMBEDDING__REVISION` | `""` | onnx only; recorded in `index_meta`. |
| `EMBEDDING__MODEL_DIR` | `./models/gte-small` | onnx only: local `model.onnx` + `tokenizer.json` (fetch with `scripts/fetch_model.py`). |
| `EMBEDDING__MAX_SEQ_LENGTH` | `512` | onnx only. |
| `EMBEDDING__POOLING` | `mean` | onnx pooling: `mean` (encoders) · `last` (decoder models) · `cls`. |
| `EMBEDDING__NORMALIZE` | `l2` | `l2` or `none`. |
| `EMBEDDING__QUERY_PREFIX` | `""` | Prepended to queries (asymmetric models: BGE/E5/Qwen). |
| `EMBEDDING__PASSAGE_PREFIX` | `""` | Prepended to passages. |
| `EMBEDDING__BATCH_SIZE` | `32` | Embed-stage / API-request batching. |
| `EMBEDDING__INJECT_HEADER_PATH` | `false` | Prepend each chunk's section-header path before embedding (part of the embedding identity — changes the fingerprint). |
| `EMBEDDING__CONTEXTUALIZE_TABLES` | `false` | Embed TABLE chunks as their header-contextualized rendering (embed-time only; also part of the identity). |
| `EMBEDDING__API_KEY` | `""` | API providers only; falls back to `OPENAI_API_KEY` / `VOYAGE_API_KEY` / `GEMINI_API_KEY`. |
| `EMBEDDING__API_BASE_URL` | `""` | Falls back to the provider's default endpoint. |
| `EMBEDDING__API_TIMEOUT` | `30.0` | Seconds. |

## `rerank` — the optional cross-encoder

Only loaded when a `cross_encoder` reranker is in the retrieval pipeline; loads lazily on first use.

| Env var | Default | Description |
|---|---|---|
| `RERANK__MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model id (for eval / result transparency). |
| `RERANK__REVISION` | `""` | |
| `RERANK__MODEL_DIR` | `./models/ms-marco-MiniLM-L-6-v2` | Local `model.onnx` + `tokenizer.json`. |
| `RERANK__MAX_SEQ_LENGTH` | `512` | |

## `llm` — the generation language model

Built only when a `GenerationEngine` is created — retrieval-only deployments never need a key.

| Env var | Default | Description |
|---|---|---|
| `LLM__PROVIDER` | `anthropic` | `anthropic` (Claude SDK) or `openai` (any OpenAI-compatible `/chat/completions` endpoint — vLLM / Together / Groq / …). |
| `LLM__MODEL` | `claude-sonnet-4-6` | |
| `LLM__API_KEY` | `""` | The key itself — avoid in files; prefer the env-var route below. |
| `LLM__API_KEY_ENV` | `""` | *Name* of the env var holding the key; `""` → the provider's standard (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). |
| `LLM__API_BASE_URL` | `""` | Falls back to the provider's default endpoint. |
| `LLM__API_TIMEOUT` | `60.0` | Seconds. |
| `LLM__MAX_TOKENS` | `1024` | Default a `Prompt` may override per call. |
| `LLM__TEMPERATURE` | `0.0` | Default a `Prompt` may override per call. |

## `database` — the two stores

| Env var | Default | Description |
|---|---|---|
| `DATABASE__DOCUMENT_URL` | `sqlite:///./rag_docs.db` | The repository: documents, chunks, the §8 retrieval index, job status. SQLite in embedded mode, Postgres (+pgvector) in distributed. |
| `DATABASE__QUEUE_URL` | `""` | The pgQueuer job queue — required for (and only used in) `MODE=distributed`. Never the same concern as `document_url`. |

## `worker` — distributed-mode tuning

| Env var | Default | Description |
|---|---|---|
| `WORKER__QUEUE_TIMEOUT_SECONDS` | `30` | How long an idle worker waits for a job before re-polling the queue (pgQueuer's dequeue timeout). Must be > 0. |
| `WORKER__CONCURRENCY` | `4` | Batches one worker process runs at once (stage compute runs in threads, off the event loop). Must be ≥ 1; scale out with more worker processes. |

## `observability`

| Env var | Default | Description |
|---|---|---|
| `OBSERVABILITY__ENABLED` | `false` | Core logic works identically with observability disabled. |
| `OBSERVABILITY__TYPE` | — | `structured_logging` (JSON-lines over stdlib logging); any other value installs the no-op adapter. |

## `components` — pipeline compositions

`Settings.components` maps names to component specs (each a `{"class_name": …}` document — see the
[component catalog](components.md)). Four keys are recognized, and defaults are filled in
automatically so a `Settings` is always self-complete:

| Key | Default |
|---|---|
| `ingestion_pipeline` | `LoadAndParse` → `Enrich` → `CleanAndNormalize` → `Chunk` → `Embed` |
| `retrieval_pipeline` | hybrid: `dense` + `sparse` retrievers, `rrf` fuser |
| `generation_pipeline` | `decomposition` reasoner (set `single_hop` for the cheap one-call path) |
| `license_policy` | `default_license` (purpose → permitted license classes; `third_party_copyrighted` never permitted) |

## Validation rules

`Settings` fails fast at construction on a `MODE`/store mismatch:

- `MODE=distributed` requires `DATABASE__QUEUE_URL` **and** a Postgres `DATABASE__DOCUMENT_URL`.
- `MODE=embedded` must **not** use a Postgres `DATABASE__DOCUMENT_URL` (SQLite is single-process —
  embedded only).

See also [`.env.example`](../../.env.example) for a commented starting point.
