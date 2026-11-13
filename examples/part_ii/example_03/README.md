# Example 03 · Cross-encoder reranking

One delta from Example 02 — a **cross-encoder reranker** (plus the `rerank` model block it needs):

```yaml
  retrieval_pipeline:
    ...
    fuser: { class_name: rrf }
    reranker: { class_name: cross_encoder }   # ← added
```

Reads the **same base store** (no re-ingest), and needs the reranker model:

```bash
python scripts/fetch_model.py --repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2
```

A cross-encoder scores the `(query, passage)` pair *together*, so it re-scores the fused shortlist far
more accurately: the paraphrase is **recovered** and the part number **kept** — both probes pass. But
it ranks *chunks*, so "all the steps in the compressor startup procedure?" returns a **fragment**:
the best chunk holds only half the section.

📖 **[Tutorial: Cross-encoder reranking](../../../docs/tutorials/part-ii/03-cross-encoder-reranking.md)**
— bi-encoder vs cross-encoder, and why fragmentation is a *representation* failure no retrieval knob
can fix.

## Run

```bash
python -m examples.part_ii.example_03.run
```

...or interactively (the breakdown gains a **`reranked`** stage with a movement `Δ` column):

```bash
python -m tarnrag.console examples/part_ii/example_03/config.yaml
tarn> explain service a rotary fluid machine before powering it up
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

→ Next: **[Example 04](../example_04)** — structure-aware chunking + auto-merge return the whole
section.
