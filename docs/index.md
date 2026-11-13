# tarn.rag documentation

**tarnrag** is a composable, DAG-based RAG library: documents in, a queryable vector index out,
optionally a grounded answer with a proof tree. This folder is the user-facing documentation,
organized by the [Diátaxis](https://diataxis.fr/) framework — pick the section that matches what you
need right now.

## [Tutorials](tutorials/index.md) — learning by doing

Step-by-step walkthroughs from zero to a working setup:

- [Getting started](tutorials/getting-started.md) — install, configure, and run your first
  ingest → retrieve → ask round trip with the Python API.
- [A console session](tutorials/console-session.md) — the same round trip in the interactive
  `tarnrag` console, no code required.
- [Part I — the fundamentals](tutorials/part-i/index.md) — a guided series over the runnable
  examples: ingestion and retrieval, a pipeline as config, comparing retrieval methods, grounded
  generation.
- [Part II — the failure-driven ladder](tutorials/part-ii/index.md) — one config knob per rung, each
  fixing the previous rung's failure.

## [How-to guides](how-to/index.md) — recipes for a task

Focused instructions for readers who know what they want:

- [Run distributed mode](how-to/run-distributed-mode.md) — Postgres + pgQueuer workers instead of
  the in-process default.
- [Swap the retrieval pipeline](how-to/swap-the-retrieval-pipeline.md) — dense-only, hybrid,
  reranked, or query-routed retrieval by editing one component spec.
- [Use hosted embedding APIs](how-to/use-hosted-embeddings.md) — OpenAI / Voyage / Gemini embedders
  in place of the local ONNX model.

## [Reference](reference/index.md) — look things up

Exhaustive and factual:

- [Settings & environment variables](reference/settings.md) — every config group, field, and
  default, plus how config files and env vars interact.
- [Component catalog](reference/components.md) — every registered component tag and the slot it
  plugs into.
- [Console commands](reference/console-commands.md) — the `tarn>` command set.

## [Explanation](explanation/index.md) — how it works and why

- [Architecture overview](explanation/architecture.md) — the three engines, the one store, and the
  invariants that hold the design together.

## Related material elsewhere in the repo

- [`examples/`](../examples/README.md) — runnable, self-contained example programs with sample corpora.
- [`doc/`](../doc) — the internal design-spec archive (functional requirements, subsystem specs,
  phase logs). The explanation pages link into it where a spec is the source of truth.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — developer setup, test commands, and conventions.
