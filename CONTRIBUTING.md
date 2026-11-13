# Contributing to tarn.rag

Thanks for contributing! This page covers the mechanics — environment, tests, conventions, and
workflow. For the architecture (and the seams to preserve when changing it), read
[`CLAUDE.md`](./CLAUDE.md) and the design specs in [`doc/`](./doc); user-facing documentation lives
in [`docs/`](./docs/index.md).

## Development setup

Python 3.12+ is required. Development happens in a conda env named `tarn.rag`, but any 3.12 venv
works:

```bash
conda create -n tarn.rag python=3.12 -y
conda activate tarn.rag
pip install -e ".[all,dev]"          # runtime extras + pytest/pytest-asyncio/pytest-cov
```

Optional, for the real-embedder test and the console/eval workflows:

```bash
python scripts/fetch_model.py        # ONNX model + tokenizer into ./models/
```

## Running the checks

```bash
python -m pytest -q                  # full suite
python -m pytest tests/ingestion -q  # a subset
ruff check tarnrag tests             # lint
```

The suite runs entirely on SQLite + the in-memory queue — **no Postgres or pgQueuer needed**. The
real ONNX embedder test is gated on the model dir existing and skips otherwise. Please keep both
properties: new tests must not require external services, and heavy backends must import lazily so
the embedded path stays light.

## Conventions

- **Types (Python 3.12):** builtin generics (`list`, `dict`, `tuple`) and `X | None` — never
  `typing.List` / `Optional`. `Iterator` / `Iterable` / `Callable` come from `collections.abc`;
  `Any` / `Literal` stay in `typing`. Use `datetime.now(UTC)`.
- **Interfaces:** ABCs (`abc.ABC` + `@abstractmethod`) over `typing.Protocol`; implementations
  inherit explicitly.
- **Pydantic v2** idioms throughout (`model_config = ConfigDict(...)`, `model_validator`).
- **Config:** new knobs go in the right `Settings` sub-model (env var `GROUP__FIELD`); pipeline
  compositions are component specs under `Settings.components`, never hard-wired.
- **Observability is optional:** core logic must work with `observability=None` — guard every
  `self.obs` call.
- Pluggable behavior (extractors, retrievers, reasoners, …) is added as a **Component** with a
  registered tag, so it is selectable from config — see the
  [component catalog](./docs/reference/components.md).
- **Docs and examples are separate trees.** [`docs/`](./docs/index.md) is prose, organized by
  [Diátaxis](https://diataxis.fr/) (tutorials · how-to · reference · explanation) — don't add folders
  named after artifact types. [`examples/`](./examples/README.md) is runnable code (an importable
  package, hence `part_i` with an underscore). A walkthrough of an example is a **tutorial** under
  `docs/tutorials/`; the example dir keeps only a short card README linking to it. Numbers published
  in docs must come from *running* the example against current code — defaults move. Full rules in
  [`CLAUDE.md`](./CLAUDE.md#documentation--examples-two-trees-two-jobs).

## Architectural ground rules

The decoupled seams documented in [`CLAUDE.md`](./CLAUDE.md) are load-bearing; in brief:

- Stages stay **pure** (no DB/queue access); worker = compute, sink = persistence, orchestrator =
  lifecycle. Don't leak one role into another.
- The queue port stays a delegating seam — pgQueuer mechanics never reimplemented on this side.
- `DATABASE__DOCUMENT_URL` and `DATABASE__QUEUE_URL` are different concerns; never conflate them.
- Shared repository logic lives in the SQLAlchemy base; dialect specifics only in the
  Postgres/SQLite hooks.
- Jobs are internal — never part of the public API surface.

## Workflow

- Branch off **`develop`**; PRs target `develop` (`main` tracks releases).
- Keep PRs focused; include tests for behavior changes and update the relevant docs
  (`docs/` for user-facing changes, `doc/` specs only when the design itself changes).
- Before pushing: `python -m pytest -q && ruff check tarnrag tests`.

## Reporting issues

Please include: what you ran, what you expected, what happened (full traceback), your mode
(embedded/distributed), and the relevant config (redact keys — only env-var *names* belong in
configs anyway).
