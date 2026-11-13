# Tutorials

Learning-oriented walkthroughs: each takes you from a clean start to a working result, explaining
just enough along the way. Do them in order if you're new.

1. [Getting started](getting-started.md) — install, configure, and run your first
   ingest → retrieve → ask round trip with the Python API.
2. [A console session](console-session.md) — the same round trip in the interactive `tarnrag`
   console, no code required.

## Guided example series

Longer walkthroughs over the runnable programs in [`examples/`](../../examples/README.md). Do them
after the two tutorials above.

3. [Part I — the fundamentals](part-i/index.md) — four steps over a three-document corpus: ingestion
   and retrieval, a pipeline as config, comparing retrieval methods with the eval harness, and
   grounded generation with a proof tree. Zero infrastructure, all offline.
4. [Part II — the failure-driven ladder](part-ii/index.md) — each rung adds one config knob, shows a
   query that works and one that fails, and the next rung repairs the failure.

Looking for something else? Recipes for a specific task live in [how-to guides](../how-to/index.md).

[← Documentation index](../index.md)
