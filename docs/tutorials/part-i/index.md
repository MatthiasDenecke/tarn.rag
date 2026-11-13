# Part I — the fundamentals

Four short tutorials over the runnable programs in [`examples/part_i/`](../../../examples/part_i).
Each one adds a single idea, and each ends with a store on disk you can poke at.

They pick up where [Getting started](../getting-started.md) leaves off: that page shows you the
`ingest → retrieve → ask` round trip; these show you *what the knobs do*.

1. [Ingestion and retrieval](01-ingestion-and-retrieval.md) — the smallest complete loop, and how to
   read a retrieval score.
2. [A pipeline from JSON](02-pipeline-from-json.md) — a pipeline is data. Change the composition
   without touching Python.
3. [Comparing retrieval methods](03-comparing-retrieval-methods.md) — dense vs sparse vs hybrid vs
   routed, scored on a labeled set.
4. [Grounded generation](04-grounded-generation.md) — question → answer + proof tree, or a refusal.

## Setup

From the repo root, once:

```bash
pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
python scripts/fetch_model.py  # download the embedding model + tokenizer (needs network)
```

Everything in Part I runs **embedded**: the whole pipeline executes in your process, over a local
SQLite file. No Postgres, no queue, no worker to start. Only tutorial 4's generation step calls out
to a network service (an LLM); the rest is fully offline.

Run the examples as modules **from the repo root** — that puts the root on the import path, so both
`examples` and `tarnrag` resolve:

```bash
python -m examples.part_i.example_01.ingestion
```

Each example writes its **own** `rag_docs.db` next to its script, so they are independent and can be
run in any order — and re-run, since ingestion upserts rather than duplicates. The stores are
gitignored; delete one and re-run its ingestion to start clean. (Set `EXAMPLES_DATA_DIR` to redirect
the stores out of the source tree.)

## The corpus

Part I uses `examples/docs/corpus-1/` — three short text files, deliberately tiny enough that you can
hold the entire index in your head and check every result by eye:

| Document | About |
|----------|-------|
| `pump-maintenance.txt` | Servicing a centrifugal pump: mechanical seal, bearing lubrication, shaft alignment. |
| `tank-inspection.txt` | Inspecting a storage tank: corrosion, shell thickness by ultrasonic testing, foundation settlement. |
| `quokka.txt` | A marsupial native to Western Australia. |

The quokka is not a joke. It is the off-topic document — the thing a query should *not* retrieve, and
the thing that makes a wrong answer visible. Watch where it lands.

Because the corpus is three documents, treat every number these tutorials print as *illustrative*,
never as a benchmark. The machinery is the lesson.

## What the examples share

Setup that every example would otherwise repeat lives in `examples/common.py`, so each script
contains only the idea it is teaching:

| Helper | What it gives you |
|--------|-------------------|
| `base_settings(db_path, **overrides)` | Default embedded/SQLite `Settings`. Override just the knob being demonstrated. |
| `example_db(__file__)` | A `rag_docs.db` next to the calling example. |
| `corpus("corpus-1")` | Path to a named corpus under `examples/docs/`. |
| `require_model()` | A friendly error if the embedding model has not been fetched. |

Two of its defaults matter enough to name:

- **`MODE='embedded'`** — the pipeline runs in-process, so `ingest` returns only when documents are
  fully chunked, embedded, and written. Every status you get back is *terminal*. (In distributed
  mode, `ingest` enqueues and returns while workers do the work — see
  [Run distributed mode](../../how-to/run-distributed-mode.md).)
- **`ID_POLICY='caller'`** — you supply stable document ids. Passing a file path makes the file's stem
  the id (`tank-inspection.txt` → `tank-inspection`), which is what turns a re-run into an upsert
  instead of a duplicate.

## One rule that will bite you

**Ingestion and retrieval must be built from the same store *and* the same `embedding` config.**

An index records an embedding *fingerprint* at ingest time, and retrieval refuses to open a store
whose fingerprint doesn't match the embedder it was handed. That is deliberate: a query embedded by a
different model than the passages is meaningless, and quietly returning nonsense is worse than
failing loudly. Reusing `base_settings(example_db(__file__))` on both sides is what keeps them in
agreement.

Note what the rule does *not* say. Retrieval never needs to know how ingestion **chunked** — that is
baked into the index. It only needs to agree on the **embedder**, which is a live contract between the
two sides. That asymmetry is why tutorial 2's retrieval script ignores the pipeline file, and why
Part II can swap retrieval pipelines freely over one store while a chunking change forces a re-ingest.

---

[← Tutorials](../index.md) · [Part II →](../part-ii/index.md)
