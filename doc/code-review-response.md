# Response to the external code review (November 2026)

The review (`CODE_REVIEW.md`, not checked in) reported **24 findings** in six work packages. Each one
was checked against the code before anything was changed. This page records the verdict for each, why,
and where the fix landed. It also lists the problems found while doing the fixes.

**Verdicts:** 15 agree, 6 agree in part (the defect is real but the report gets the location, cause, or
fix wrong), and 3 disagree (the code described doesn't exist, or already does what's asked). Several
"agree" items needed a different fix than the one proposed. Those are explained below.

The fixes are a stack of PRs. Merge them in order, because each one builds on the last:

| PR | Scope |
|---|---|
| #131 | Upgrade **bausatz to 0.3** (typed `Config` children); `sqlalchemy[asyncio]` dependency fix; the repo-wide ruff fix (LOW-05), at the base so every PR passes the Ruff check |
| #132 | WP-1: CRIT-01, MED-01, MED-02 (LOW-04 rejected) |
| #133 | WP-2: CRIT-02, HIGH-01 |
| #134 | WP-3a: HIGH-02, HIGH-03, HIGH-04 |
| #135 | WP-3b: HIGH-05, MED-08 (LOW-03 rejected) |
| #136 | WP-4a: HIGH-06, HIGH-08 |
| #137 | WP-4b: HIGH-07, MED-03, MED-04 |
| #138 | WP-5: MED-05, MED-06, MED-07 (LOW-02 rejected) |
| #139 | WP-6: MED-09, LOW-01, and this page |

## Verdicts

| ID | Verdict | Notes |
|---|---|---|
| CRIT-01 | Agree | No production code ever set `Purpose.GENERATION_GROUNDING`. `GenerationPipeline.answer` now marks every retrieval `grounds_generation`, and every reasoner sub-query carries the flag. It is **additive**: it requires `ai_grounding_allowed`, and the license policy intersects the caller's purpose classes with `GENERATION_GROUNDING`'s. An absent purpose fails closed. (The first version replaced the purpose, which could *widen* a narrowed map; this was caught in review of #132.) |
| CRIT-02 | Agree | The stage was dead code. It now cleans the `StructuredDocument` through an offset map (all spans remapped, code and tables kept verbatim) and stores the cleaned text again. That store rewrites the document row only, so a Clean job replayed after its Chunk and Embed jobs leaves their chunks and vectors in place. The default stage order is unchanged, because reordering would rename the positional DAG nodes that in-flight distributed jobs refer to. |
| HIGH-01 | Agree | Section parents are no longer embedded or sparse-indexed, and every search filters to `level = 0`. That filter also covers indexes built before the change. A parent doesn't get an Embed job either. Re-ingesting an index built before the change refreshes its BM25 statistics, which the parents' old rows skew. |
| HIGH-02 | Agree | Job rows are cleared when a document is resubmitted, and queued or processing jobs now report `in_progress`. A job that never reaches the queue leaves no `queued` row behind, and a resubmission that can't be enqueued reports `failed`. In distributed mode, a job from an earlier run that is still executing can write its row after the clear; closing that needs a run id on `job_status` (follow-up). |
| HIGH-03 | Agree | `store_chunks` replaces a document's chunks, and chunk ids are content-addressed (`uuid5`), so a retried job produces the same ids. A `source_id` repeated in one call is ingested once (last occurrence), and embedded ingest calls run one after the other, so two runs of one document never share a batch. |
| HIGH-04 | Agree | Reproduced on Postgres 16 (79 `IntegrityError`s in 10 trials). It is now an atomic `INSERT … ON CONFLICT DO UPDATE`. |
| HIGH-05 | Partly | **AND vs OR (agree):** both dialects now build from the same parse. **`postgres://` URLs (agree):** fixed, along with `postgresql+psycopg2://`, which the review missed. **`english` stemming (disagree):** stemming is a deliberate recall gain on Postgres; the real gap was AND vs OR. **ivfflat recall (partly):** the pipelines always pass a filter, which already scans all lists exactly, and since #133 every Postgres dense search is exact. Switching to HNSW is a separate, benchmarked change, because HNSW's default `ef_search` would silently cut the over-fetch backfill short. |
| HIGH-06 | Agree | Stages now run in `asyncio.to_thread`. (The "`time.sleep` retries" detail is wrong; there are none.) Concurrent batches then raced the lazy builds of the ONNX session, `EmbedStage`'s embedder, the API embedder's HTTP client and docling's converter; each now builds once under a lock, and docling conversions take turns. |
| HIGH-07 | Agree | All six points hold. The common root cause is that `Engine.aclose` closed injected stores, so ownership is now explicit: an engine releases only what it built. #137's review found one more: `IngestionEngine.create` leaked the store and embedder when the queue failed to connect; it now releases them on any failure. |
| HIGH-08 | Agree | Settings are now wired to the worker. **The proposed `max_concurrent_tasks=concurrency` would stop the worker loop**: pgQueuer requires the value to be at least twice the batch size, so `batch_size` is derived from it and a semaphore keeps the cap exact. |
| MED-01 | Partly | License and grounding flags are set per document, so a parent always shares them with its leaves. The real leak is **method scope**: a method's chunk set can include leaves without their section parent. Parents now pass `ChunkFilter.admits`, and merging repeats up the section tree. |
| MED-02 | Agree | The router now classifies a copy of the query, and a nested router runs its own classifier. |
| MED-03 | Agree | API embedders now apply the prefixes, and the API and hash fingerprints record prefixes and embed-time flags (only when set). The claim that `_ApiEmbedder._identity()` already includes the prefixes is wrong; it didn't. An API index built with a prefix or flag set must be re-ingested once after upgrading (`use-hosted-embeddings.md`). |
| MED-04 | Agree (minor) | The tokenizer hash is computed once per instance, and the error now names the fix. It stays a `FileNotFoundError`, because there is no `ConfigurationError` in the codebase. |
| MED-05 | Agree | Markdown first-column cells are now row headers, a table may directly follow a paragraph, and HTML cells are placed on the real grid (also fixing nested tables and malformed spans). |
| MED-06 | Agree | Verdicts are parsed strictly, and anything unreadable is `UNCERTAIN` (the pipeline's lenient default keeps "no spurious refusal"). An uncited step is `UNGROUNDED`, and `"steps": []` no longer produces made-up citations. |
| MED-07 | Partly | `_match_columns` doesn't exist; columns are matched by year. The same defect sits in **`_best_row`**: the article "a" counted toward a "Plan A" label. That's fixed there. |
| MED-08 | Partly | The double read in `hydrate` is real; it's now one query on `chunks`. The 999-variable SQLite limit applies only before 3.32 (now 32 766), but the subquery delete is strictly better, so it went in. |
| MED-09 | Agree | The tutorial config now sets `ID_POLICY: caller` and explains it. |
| LOW-01 | Agree | `shlex.split` (non-POSIX on Windows), with an error message for an unbalanced quote. |
| LOW-02 | **Disagree** | Neither `_WARNED_NO_CRYPTO` nor `_extract_paragraphs` exists. `PdfTextExtractor` uses pdfplumber's line boxes as-is and never divides by page height. |
| LOW-03 | **Disagree** | `InMemoryJobQueue` already keeps `dead_letters`, and the worker records `failed` before re-raising. `get_queue_stats()` doesn't exist. |
| LOW-04 | **Disagree** | Neither bridge retriever builds a sub-`Query`. Both pass `ctx.filter_for(query)`, which includes the scope, to every `dense_knn` call. |
| LOW-05 | Agree | Fixed at the base of the stack (#131), so every PR passes the Ruff check: `ruff check .` now passes. Intentional broad `except`s keep a `noqa` with the reason; everything else was fixed. Pinning ruff's version in CI would stop new default rules from failing builds unexpectedly. |

## Found while fixing

- **A fresh install was broken.** `bausatz>=0.1` resolves to 0.3, whose 0.2 API changes failed 99
  tests, and SQLAlchemy 2.1 no longer installs `greenlet`, so the async repository couldn't even be
  imported. Fixed in #131.
- **ONNX lazy-load race.** Once stages ran in threads, two first calls could each build the session,
  and the cross-encoder made its session visible before its input names were set. Both models now use
  a lock and set the session last (#136).
- **Embedder mismatch.** An embedder passed to `IngestionEngine.create` was used to stamp `index_meta`,
  but the Embed stage built its own from `Settings`. Fixed by binding the shared embedder into the
  stage (#137).
- **Partial staged files.** A stream that broke mid-copy left a partial file in `UPLOAD_DIR` (#137).
- **A regression in this work, caught and fixed.** The first version of the AutoMerger's "covered by
  an ancestor" check compared `source_element_ids`, which are never persisted, so it never fired on
  real data. It now compares persisted char spans. An end-to-end test over really ingested chunks
  catches it (#132).
- **Not fixed: distributed retries.** The distributed how-to and the code comments say a job whose
  handler raises is requeued. With pgQueuer's default executor it isn't: the job is logged as
  `exception` and deleted (`on_failure="delete"`). Only a crashed worker's job comes back, once its
  heartbeat goes stale. The status correctly shows `failed`, but nothing retries it. Making it retry
  needs a policy decision (for example pgQueuer's `DatabaseRetryEntrypointExecutor` with a maximum
  attempt count and backoff), so it's left for a follow-up.

## How the fixes were checked

Every fix has a regression test, and most were confirmed to fail on the old code. The SQLite suite
runs everywhere (537 passing at the top of the stack). The Postgres and pgQueuer tests that CI
otherwise runs were run against a throwaway pgvector 0.7.4 / Postgres 16 cluster (11/11 passing).
