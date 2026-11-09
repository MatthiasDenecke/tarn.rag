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

Worker tuning lives under `WORKER__*` (`WORKER__CONCURRENCY`, default 4;
`WORKER__QUEUE_TIMEOUT_SECONDS`, default 30). pgQueuer owns the queue mechanics — SKIP LOCKED
dispatch, retries, NOTIFY wake-ups, dead-lettering. A worker that fails a batch re-raises, and the
queue requeues the job; downstream jobs are only enqueued after upstream results are persisted, so
a crash never loses or duplicates work (re-ingesting is idempotent by document id).

## Notes

- Retrieval and generation read the Postgres store directly; only ingestion is queued.
- The test suite never needs this setup — it runs entirely on SQLite + the in-memory queue.
- Full design: [`doc/FUNCTIONAL_REQUIREMENTS.md`](../../doc/FUNCTIONAL_REQUIREMENTS.md) and the
  [architecture overview](../explanation/architecture.md).
