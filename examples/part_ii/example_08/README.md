# Example 08 · Multi-hop reasoning (decomposition)

Two deltas from Example 07 — its **own store**, and a multi-hop reasoner:

```yaml
  database:
    document_url: sqlite:///./examples/part_ii/store/multihop.db            # ← own store
  ...
    reasoner: { class_name: decomposition, max_subquestions: 4, top_k: 8 }  # ← was single_hop
```

*"Which oil grade does the TX-200 require?"* spans two documents — `compressor-models` (the TX-200 is a
**GMV frame**) and `lubrication-spec` (GMV frames take **ISO VG 68**) — and neither holds the chain.
This store also has four near-identical frame specs (Ariel → VG 100, Waukesha → VG 46, Superior → VG
150, Clark → VG 32), so the GMV spec is **crowded out of the top-4**: `single_hop` never sees it and
honestly answers *"not specified"*. `decomposition` retrieves per sub-question, so the second hop lands
the bridge → **ISO VG 68**, citing both docs.

The MISS is visible **model-free** in the `explain` — no reasoner can read a passage retrieval never
returns. It's a *reachability* failure, not a reading failure.

⚠️ The distractors are what make the bridge a bridge. On the plain 12-doc corpus, `single_hop`,
`decomposition` *and* `iterative` all answer correctly — multi-hop only pays when the bridge document is
genuinely out of reach. It also costs several LLM calls where one hop paid one.

📖 **[Tutorial: Multi-hop reasoning](../../../docs/tutorials/part-ii/08-multi-hop.md)** — why the
distractors are necessary, decomposition vs iterative, and what the extra hops cost.

## Run

Needs `pip install '.[openai]'` and a key in **`OPENAI_LLM_KEY`** (or a repo-root `.env`).

```bash
python -m examples.part_ii.example_08.run
```

...or interactively — this example has its **own** store, so build it first (two ingests):

```bash
python -m tarnrag.console examples/part_ii/example_08/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> ingest examples/docs/corpus-2-frames
tarn> status                                          # 16 documents · 24 chunks
tarn> explain which oil grade does the TX-200 require?   # the GMV spec is ABSENT from the candidates
tarn> ask which oil grade does the TX-200 require?       # ...yet the answer arrives: ISO VG 68
```

→ Next: **Example 09** — the answerability gate: refuse *before* the read, not after the guess.
