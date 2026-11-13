# 0 · Ingest the corpus

> **Setup.** Build the base store every later rung queries — then never ingest again.
> **Config:** `examples/part_ii/example_00/config.yaml` · **Offline**

Rungs 01, 02, 03, 05, 06, and 07 all read *this* store. They differ only in configuration. Ingesting
once and then swapping config is what makes the ladder honest: when a rank moves, it moved because of
the knob you turned, not because the data changed underneath you.

## The baseline

`config.yaml` specifies the **complete** system — ingestion, retrieval, and generation — so every
console command (`ingest`, `explain`, `ask`) is fully defined from the start. Later rungs vary one
pipeline at a time from this baseline, and the diff between two configs is that rung's entire lesson.

The ingestion DAG, stated explicitly:

```
each .md file → LoadAndParse (markdown) → CleanAndNormalize → Chunk (recursive 400/80) → Embed → store
```

The 12 Markdown documents of `corpus-2` become **~19 overlapping chunks**, embedded with `gte-small`
into one SQLite file (`examples/part_ii/store/base.db`).

Note the chunker: `recursive` produces a **flat** set of overlapping windows — no section tree, no
parent chunks. That is the right default for the retrieval rungs that follow. (The structure-aware
chunker, which builds section parents, would put parent/leaf duplicates in every result table and
clutter a plain dense-vs-sparse comparison. It arrives in [rung 04](04-structure-aware-chunking.md),
where it is the point.)

## Run it

The console is the hero path — the config *is* the example:

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> status
tarn> explain how do I prevent a pump from cavitating?
```

Or as a script, which ingests and then prints one `explain` breakdown:

```bash
python -m examples.part_ii.example_00.run
```

Ingestion is idempotent — re-running refreshes the store rather than duplicating it.

## The "no defaults" rule

Every value in `config.yaml` is set explicitly. Nothing inherits a library default.

This looks like ceremony until it saves you. The file alone tells you exactly what the pipeline does,
with no need to cross-reference the library's current defaults — and the results cannot drift when
those defaults change. That is not hypothetical: while these pages were being written, the library's
default retrieval pipeline changed from dense-only to hybrid, and every number in Part II was
untouched, because no Part II config was relying on a default.

An example whose behaviour depends on a default is an example that silently rots.

## If you want an LLM

Retrieval and `explain` need no LLM — the reader is built lazily, on the first `ask`. To use `ask`
(rungs [06](06-minimal-generation.md) and [07](07-grounding-and-abstain.md)):

```bash
pip install '.[openai]'
```

and put your key in **`OPENAI_LLM_KEY`** — either exported, or in a `.env` at the repo root, which
the console and the example runner both load. The config names *the variable*, never the key itself,
so no secret ever lives in a config file.

## Next

**[1 · Dense-only retrieval →](01-dense-only-retrieval.md)** — query this store the simplest way there
is, and meet the ladder's first failure.

---

[← Part II](index.md)
