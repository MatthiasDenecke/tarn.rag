# Example 02 · Hybrid retrieval (dense + sparse + RRF)

One delta from the base config — a **sparse (BM25)** retriever beside the dense one, fused with
**RRF** (`diff examples/part_ii/example_00/config.yaml examples/part_ii/example_02/config.yaml`):

```yaml
  retrieval_pipeline:
    retrievers:
      - class_name: dense
      - class_name: sparse        # ← added
    fuser: { class_name: rrf }    # ← was identity
```

Reads the **same base store** (no re-ingest). The two probes from Example 01 **swap**: the part number
`XQ-9920-A` now HITs at rank 1, while the paraphrase falls out of the top 4 — fusion also imports
sparse's confidently wrong opinions.

📖 **[Tutorial: Hybrid retrieval](../../../docs/tutorials/part-ii/02-hybrid-retrieval.md)** — why RRF
fuses ranks rather than scores, and why fusion is never free.

## Run

```bash
python -m examples.part_ii.example_02.run
```

...or interactively (each `explain` now prints **two** per-retriever tables):

```bash
python -m tarnrag.console examples/part_ii/example_02/config.yaml
tarn> explain XQ-9920-A
tarn> explain service a rotary fluid machine before powering it up
```

→ Next: **[Example 03](../example_03)** — a cross-encoder reranker recovers the paraphrase *and* keeps
the part number.
