# Example 00 · Ingest the shared corpus

**Setup step.** Builds the `base` store (`store/base.db`) — the SQLite index every later example
queries. Ingest once here; the retrieval and generation steps then only swap their *config* over this
same store.

`config.yaml` specifies the complete baseline — ingestion **and** retrieval **and** generation — with
every value explicit, so nothing relies on a library default.

📖 **[Tutorial: Ingest the corpus](../../../docs/tutorials/part-ii/00-ingest-the-corpus.md)** — the
ingestion DAG, the "no defaults" rule, and why it matters.

## Run

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml   # the hero path
tarn> ingest examples/docs/corpus-2
tarn> status
tarn> explain how do I prevent a pump from cavitating?
```

...or as a script (ingests, then prints one `explain` breakdown):

```bash
python -m examples.part_ii.example_00.run
```

Needs the embedding model (`python scripts/fetch_model.py`, once). `ask` additionally needs
`pip install '.[openai]'` and a key in **`OPENAI_LLM_KEY`** (export it, or use a repo-root `.env`).

→ Next: **[Example 01](../example_01)** — dense-only retrieval, and its first failure.
