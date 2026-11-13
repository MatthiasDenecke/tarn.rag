# Run distributed mode

Embedded mode (the default) runs the whole pipeline in-process over SQLite. Distributed mode
enqueues ingestion jobs to **pgQueuer** over Postgres and processes them in one or more separate
worker processes — for throughput, isolation, or multi-host deployments.

## 1. Install the distributed backends

```bash
pip install "tarn-rag[postgres,queue]"    # asyncpg + pgvector, pgqueuer
```

The Postgres document store uses the `pgvector` extension for dense retrieval; make sure it is
available in your database (`CREATE EXTENSION vector;`).

## 2. Configure the two databases

Distributed mode uses **two** database URLs — never conflate them:

```bash
MODE=distributed
DATABASE__DOCUMENT_URL=postgresql://user:password@localhost:5432/tarn_rag_docs   # documents/chunks/index
DATABASE__QUEUE_URL=postgresql://user:password@localhost:5432/tarn_rag_queue     # the pgQueuer job queue
```

Any Postgres URL form works for `DATABASE__DOCUMENT_URL` (`postgresql://`, or the `postgres://` alias
many hosting providers hand out); the repository always drives it through asyncpg. A libpq
`?sslmode=…` parameter is translated to asyncpg's `ssl=…` (same mode names, e.g. `require`).

`Settings` fails fast on a mismatch: `MODE=distributed` requires a Postgres `document_url` **and** a
`queue_url`; `MODE=embedded` requires SQLite (it is single-process by design). You cannot silently
run distributed on the SQLite default.

If you ingest byte streams (`ingest_streams`), `UPLOAD_DIR` is where they are staged for the
workers — in a multi-host deployment it must be a shared volume visible to producers and workers
alike.

## 3. Run workers

The producer side is unchanged — `TarnRag` / `IngestionEngine.create()` reads `MODE` and enqueues
instead of processing inline. Consume with one or more worker processes:

```bash
python run_worker.py
```

Worker tuning lives under `WORKER__*`: `WORKER__CONCURRENCY` (default 4) is how many batches one
worker process runs at once, and `WORKER__QUEUE_TIMEOUT_SECONDS` (default 30) how long an idle worker
waits before re-polling. Stage compute (parsing, embedding) runs in threads, so a busy worker still
answers the queue's heartbeats. pgQueuer owns the queue mechanics — SKIP LOCKED
dispatch, retries, NOTIFY wake-ups, dead-lettering. A worker that fails a batch re-raises, and the
queue requeues the job; downstream jobs are only enqueued after upstream results are persisted, so
a crash never loses or duplicates work (re-ingesting is idempotent by document id).

### Sizing `WORKER__CONCURRENCY`

- **Local ONNX embedding is already parallel.** ONNX Runtime sizes its thread pool to every core, so
  one Embed batch keeps the machine busy on its own. At `WORKER__CONCURRENCY=4`, four of them compete
  for the same cores: about four times as many runnable threads as cores, for little extra throughput.
  For workers that embed locally, keep concurrency at 1–2 and add worker processes or hosts to scale.
  Higher values pay off when stages mostly wait, as with an embedding API.
- **Docling extractions take turns.** With the opt-in `docling` extractor, one worker process converts
  one PDF at a time, whatever the concurrency, because docling doesn't promise that one converter is
  safe to share between threads. Docling spreads each conversion over several threads itself.
- **A single-slot worker holds one job in reserve.** pgQueuer requires a worker to allow at least twice
  as many in-flight jobs as it claims per poll. So at `WORKER__CONCURRENCY=1` a worker claims 2 jobs and
  runs 1, and the second waits, claimed, until the first finishes; no other worker can take it
  meanwhile. Higher concurrency claims no more jobs than it runs.

## Notes

- Retrieval and generation read the Postgres store directly; only ingestion is queued.
- The test suite never needs this setup — it runs entirely on SQLite + the in-memory queue.
- Full design: [`doc/FUNCTIONAL_REQUIREMENTS.md`](../../doc/FUNCTIONAL_REQUIREMENTS.md) and the
  [architecture overview](../explanation/architecture.md).
