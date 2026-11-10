# Swap the retrieval pipeline

Retrieval is composed from config-driven components under `Settings.components.retrieval_pipeline`.
Comparing retrieval methods means swapping this spec — no code changes. Specs are shown in YAML
here; JSON works identically.

## The default: hybrid

If you configure nothing, you get dense KNN + sparse BM25 fused with reciprocal-rank fusion —
measured to never lose to dense-only on any evaluated segment (see `doc/phases.md`):

```yaml
components:
  retrieval_pipeline:
    class_name: retrieval_pipeline
    retrievers: [{class_name: dense}, {class_name: sparse}]
    fuser: {class_name: rrf}
```

## Dense-only

Drop the sparse retriever and fuse with `identity`:

```yaml
components:
  retrieval_pipeline:
    class_name: retrieval_pipeline
    retrievers: [{class_name: dense}]
    fuser: {class_name: identity}
```

## Add a cross-encoder reranker

For quality-sensitive deployments, rerank the fused candidates with a local ONNX cross-encoder
(measured: table source-hit 0.853 → 0.926 on TAT-QA, at a cost of ~seconds/query of CPU):

```yaml
components:
  retrieval_pipeline:
    class_name: retrieval_pipeline
    retrievers: [{class_name: dense}, {class_name: sparse}]
    fuser: {class_name: rrf}
    reranker: {class_name: cross_encoder, top_n: 20}
```

The reranker model is configured under `settings.rerank` (default
`cross-encoder/ms-marco-MiniLM-L-6-v2`); fetch it once with `scripts/fetch_model.py`. It loads
lazily on first use. An `llm_judge` reranker also exists for LLM-scored reranking.

## Add auto-merge

The optional `merger` slot recombines sibling chunks into their parent when enough of them are
retrieved together, returning coherent larger passages. It repeats up the section tree (merged
subsections can merge into *their* section), and a parent is only substituted when the query may
see it (same license, scope, and grounding filter the retrievers apply):

```yaml
    merger: {class_name: auto_merge}
```

## Route by query type

`routing_retrieval_pipeline` dispatches to different pipelines per classified query type — a
`QueryClassifier` (`generic`, `intent`, or `structural`) decides, and each `query_type` gets its own
pipeline spec. Use it when one composition can't serve all query shapes (e.g. table lookups vs.
prose questions).

## LLM-assisted retrievers

Beyond `dense` and `sparse`, two bridge retrievers rewrite the query before dense search: `hyde`
(embeds a hypothetical generated answer) and `multi_query` (fans out reformulations). Both need the
generation LLM configured.

## Verify what you built

The console's `explain` command shows each retriever's candidates before fusion and the ranking at
every stage — the quickest way to confirm a spec change does what you expect. The full slot-by-slot
list of components is in the [component catalog](../reference/components.md).
