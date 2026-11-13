# Example 01 · Dense-only retrieval

A single **dense** retriever — vector KNN over the embedded chunks. Reuses the base store *and*
config from [Example 00](../example_00); there is no config delta yet (the first one is Example 02).

Dense matches by **meaning** — so a paraphrase with no shared words HITs (rank 3), while the opaque
part number `XQ-9920-A` MISSes (the right doc lands at rank 5, outside the top 4): a bare identifier
has no meaning to embed.

📖 **[Tutorial: Dense-only retrieval](../../../docs/tutorials/part-ii/01-dense-only-retrieval.md)** —
both probes, the `explain` breakdown, and why no better embedding fixes this.

## Run

```bash
python -m examples.part_ii.example_01.run
```

...or `explain` both queries interactively over the base config:

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml
tarn> explain service a rotary fluid machine before powering it up
tarn> explain XQ-9920-A
```

→ Next: **[Example 02](../example_02)** — add sparse (BM25) + RRF. The part number starts hitting.
