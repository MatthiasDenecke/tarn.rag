# tarnrag examples

Runnable, self-contained example programs. **The prose lives in the docs** — these directories hold the
code, the configs, and the sample corpora.

| | | |
|---|---|---|
| **Part I** (`part_i/`) | Four steps over a three-document corpus: ingestion + retrieval, a pipeline as config, comparing retrieval methods, grounded generation. Zero infra, all offline (bar the last generation step). | 📖 **[Tutorials: Part I](../docs/tutorials/part-i/index.md)** |
| **Part II** (`part_ii/`) | The failure-driven ladder: each rung adds one config knob, shows a query that works and one that **fails**, and the next rung repairs it. YAML configs, console-first. | 📖 **[Tutorials: Part II](../docs/tutorials/part-ii/index.md)** |

Each `part_ii/example_NN/` also carries a short README with its config delta and run commands.

## Setup (once, from the repo root)

```bash
pip install -e ".[onnx]"       # the tarnrag package + the ONNX embedding runtime
python scripts/fetch_model.py  # the embedding model + tokenizer (needs network)
```

## Running an example

Examples are a Python package; run them as modules **from the repo root**:

```bash
# Part I
python -m examples.part_i.example_01.ingestion    # ingest the sample corpus into a local SQLite store
python -m examples.part_i.example_01.retrieval    # query that store
python -m examples.part_i.example_02.ingestion    # same, pipeline configured from pipeline.json
python -m examples.part_i.example_03.evaluation   # compare retrieval methods on a labeled set
python -m examples.part_i.example_04.generation   # question -> grounded answer + proof tree

# Part II — build the store once, then each rung is a config swap
python -m examples.part_ii.example_00.run         # ingest corpus-2 into the base store
python -m examples.part_ii.example_02.run         # hybrid retrieval (dense + sparse + RRF)
```

Running with `-m` puts the repo root on the import path, so `examples` and `tarnrag` both resolve — no
`PYTHONPATH` or `sys.path` tweaks. (Directory names use underscores because they are **module names**.)

Generation needs an LLM key; everything else runs offline. Part I's example 04 uses Anthropic
(`ANTHROPIC_API_KEY`) and previews what it *would* send the model if no key is set; Part II's configs
name **`OPENAI_LLM_KEY`** and need `pip install '.[openai]'`. A config names the *variable*, never the
key itself — a repo-root `.env` is loaded automatically.

## Interactive console

Part II is console-first: each example *is* a config you can drive interactively.

```bash
python -m tarnrag.console examples/part_ii/example_02/config.yaml   # a Part II rung
python -m tarnrag.console examples/console.config.json              # a general-purpose config
```

Commands: `ingest <path>` (file or directory; re-ingesting replaces), `docs`, `status`, `delete <id>`,
`retrieve <query>`, `explain <query>` (the full scoring breakdown — per-retriever candidates, fusion,
rerank, merge, with a movement column) and `ask <query>` (answer + proof tree). See the
[console reference](../docs/reference/console-commands.md).

The console is only a UI; the work lives in `tarnrag.TarnRag`, the facade over the three engines, which
you can drive directly. Every call returns an `Outcome` — a `value` plus a `report` of non-fatal issues,
so nothing is silently skipped:

```python
from tarnrag import TarnRag

async with TarnRag("examples/console.config.json") as tarn:
    ingested = await tarn.ingest(["examples/docs/corpus-1", "typo.txt"])
    for issue in ingested.report.issues:
        print(f"{issue.severity.value}: {issue.subject}: {issue.message}")  # warning: typo.txt: not found
    answer = await tarn.ask("How should I service a pump before restarting it?")
    print(answer.value.answer)
```

## Layout

```
examples/
├── common.py             # shared helpers (settings factory, corpus paths, model check)
├── console.config.json   # a general-purpose console config
├── docs/                 # sample corpora, ingested by path
│   ├── corpus-1/         # 3 short docs — Part I
│   └── corpus-2/         # 12 engineered Markdown docs — Part II
├── part_i/example_NN/    # scripts + JSON pipeline specs
└── part_ii/
    ├── PLAN.md · CORPUS.md   # the ladder's design notes; why each corpus doc exists
    ├── _runner.py            # shared harness: run the good/bad cases, render the breakdown
    └── example_NN/           # config.yaml + run.py + README (the card)
```

## Adding things

- **A corpus:** drop files in `examples/docs/<name>/` and select it with `corpus("<name>")`. Put design
  notes *outside* the corpus directory — ingesting a directory ingests every file in it.
- **A Part I example:** create `examples/part_i/example_NN/` with an `__init__.py` and your script(s),
  importing what you need from `examples.common` (`base_settings`, `example_db`, `corpus`,
  `require_model`). Override only the knob you are demonstrating.
- **A Part II rung:** copy the nearest `config.yaml`, change **one** thing, and keep the config fully
  explicit — no reliance on library defaults, so results can't drift when a default changes.

An example's ingestion and retrieval **must** build from the same store *and* embedding config —
retrieval validates an embedding fingerprint before opening the index.

Generated stores (`*.db`) and `__pycache__/` are gitignored.
