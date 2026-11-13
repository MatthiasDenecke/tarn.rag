# 3 · Cross-encoder reranking

> **Best of both — and a new problem: the winning chunk is only half the answer.**
> **Config:** `examples/part_ii/example_03/config.yaml` · reads the base store (no re-ingest)

One change from [rung 02](02-hybrid-retrieval.md): a **cross-encoder reranker** on the retrieval
pipeline (plus the `rerank` model block it needs).

```yaml
  retrieval_pipeline:
    ...
    fuser: { class_name: rrf }
    reranker: { class_name: cross_encoder }   # ← added
```

It needs the reranker model, fetched once:

```bash
python scripts/fetch_model.py --repo Xenova/ms-marco-MiniLM-L-6-v2 --dest ./models/ms-marco-MiniLM-L-6-v2
```

## Bi-encoder vs cross-encoder

This is the distinction the whole rung turns on.

A **bi-encoder** (what dense retrieval uses) embeds the query and the passage *separately*, into
independent vectors, and compares them by distance. That independence is what makes it fast: every
passage was embedded once, at ingest time, and a query is one vector lookup against an index. It is
also what makes it approximate — the passage's vector was computed without any knowledge of your
query.

A **cross-encoder** feeds the `(query, passage)` pair through the model *together*, so every word of
the query can attend to every word of the passage. It judges relevance far better. It is also far too
slow to run over a corpus — you would be doing a full model forward pass per passage, per query.

The resolution is to use each where it belongs: **retrieve** with the fast approximate methods, then
**rerank** only the handful of candidates they produced. Cheap over the corpus, accurate over the
shortlist.

## Best of both

| Query | dense (01) | hybrid (02) | **rerank (03)** |
|-------|------------|-------------|-----------------|
| paraphrase *"rotary fluid machine…"* | ✅ | ❌ | ✅ **recovered** |
| `XQ-9920-A` (part number) | ❌ | ✅ | ✅ **kept** |

Both probes pass — the first time in the ladder.

Run `explain service a rotary fluid machine before powering it up` and watch the **`reranked`**
stage's `Δ` column: `pump-maintenance` jumps **up** (`▲`) from where RRF had buried it. The
cross-encoder reads the query against the passage, recognizes it as the real answer, and overrules
the fusion. Rung 02's regression is undone — while `explain XQ-9920-A` keeps `pump-maintenance` at
rank **1**, carried straight through the rerank.

This is why reranking is the highest-leverage single addition to a retrieval stack. It doesn't
replace your retrievers; it *corrects* them, and it does so with a model that can actually read.

## The new failure: fragmentation

Now ask for something that spans a whole section:

```text
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

The top result is a short (~300-character) `compressor-startup` chunk. Read its text: it has the first
steps — set the valves, pre-lubricate — and **not the rest**. There is no "loading sequence". The
gold phrase is not in the top-1 result at all.

Nothing went wrong in the ranking. The reranker picked the best chunk available, and the best chunk
available is *half the procedure*, because `recursive` chunking sliced a coherent section into
overlapping windows without regard for where the section began or ended.

**The reranker ranks chunks. It cannot rank a section that was never a chunk.**

That is a failure of the *representation*, not of retrieval — and it is the one thing so far that no
retrieval-side knob can fix. Every rung until now swapped a retrieval component over the same index.
This one requires changing what is in the index.

## Run it

```bash
python -m examples.part_ii.example_03.run
```

Or interactively (the cross-encoder loads lazily, on the first query):

```bash
python -m tarnrag.console examples/part_ii/example_03/config.yaml
tarn> explain service a rotary fluid machine before powering it up
tarn> explain XQ-9920-A
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

The breakdown now shows the `fused` → **`reranked`** → `final` stages with a movement (`Δ`) column.
For contrast, run the same paraphrase under rung 02's config (which has no reranker) and watch
`pump-maintenance` sit far down the list.

## Next

**[4 · Structure-aware chunking →](04-structure-aware-chunking.md)** — chunk on the document's
headings instead of on character counts, so a section can come back whole.

---

[← Part II](index.md)
