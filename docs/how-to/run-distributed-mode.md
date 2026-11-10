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
many hosting providers hand out); the repository always drives it through asyncpg.

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
dispatch, retries, NOTIFY wake-ups, dead-lettering.

**Failures are retried.** A job whose stage (or its persistence) fails goes back on the queue with an
exponential backoff — `WORKER__RETRY_DELAY_SECONDS` (default 1 s), doubling per attempt, capped at
`WORKER__RETRY_MAX_DELAY_SECONDS` (default 300 s) — until it has run `WORKER__MAX_ATTEMPTS` times
(default 3). While a retry is pending the document's status is `in_progress`; only the last failed
attempt marks it `failed`. A job that exhausts its attempts is **kept** in pgQueuer's queue table with
status `failed` — the dead letter — so it can be inspected and re-queued (pgQueuer's
`Queries.list_failed_jobs` / `Queries.requeue_jobs`) once the cause is fixed. A worker that *crashes*
mid-job is recovered separately: its heartbeat goes stale and another worker picks the job up.
Downstream jobs are only enqueued after upstream results are persisted, and every write is idempotent,
so a retry never duplicates work.

## Notes

- Retrieval and generation read the Postgres store directly; only ingestion is queued.
- The test suite never needs this setup — it runs entirely on SQLite + the in-memory queue.
- Full design: [`doc/FUNCTIONAL_REQUIREMENTS.md`](../../doc/FUNCTIONAL_REQUIREMENTS.md) and the
  [architecture overview](../explanation/architecture.md).
