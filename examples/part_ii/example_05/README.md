# Example 05 · Query routing (Act A capstone)

One delta from the base config — the `retrieval_pipeline` becomes a `routing_retrieval_pipeline`:

```yaml
  retrieval_pipeline:
    class_name: routing_retrieval_pipeline
    classifier: { class_name: structural }
    routes:
      lexical:  { ... retrievers: [sparse] }   # exact terms / identifiers
      semantic: { ... retrievers: [dense]  }   # paraphrases
    default:    { ... dense + sparse + rrf }   # unknown → hybrid
```

Reads the **same base store**. Scored on a labeled set (`evalset.yaml`), dense, sparse, and hybrid all
tie at `0.875` hit@k overall — but segmented by query type, dense owns semantic (1.00 / 0.75) and
sparse owns lexical (0.75 / 1.00). Routing hits **1.00 on both**, beating every fixed pipeline. The
`StructuralQueryClassifier` reproduces the labels with none of its own (7/8).

⚠️ **Routing is an alternative to Example 03's reranker, not a layer on top of it.** The S1 sweep
(`doc/phases.md`) measured `ROUTED+CE` as digit-identical to `HYBRID+CE` — the cross-encoder subsumes
routing's gain, so routing under a reranker is pure cost. It pays in **no-reranker** deployments.

📖 **[Tutorial: Query routing](../../../docs/tutorials/part-ii/05-query-routing.md)** — how an
aggregate metric hid a free win, why the scoreboard routes on gold labels (oracle) while the
classifier is measured separately, and where routing actually earns its place.

## Run

```bash
python -m examples.part_ii.example_05.run    # scoreboard + classifier check + two routed explains
```

...or interactively (the breakdown announces the decision on a `routed:` line):

```bash
python -m tarnrag.console examples/part_ii/example_05/config.yaml
tarn> explain XQ-9920-A                                          # → route 'lexical', sparse only
tarn> explain how do I keep a big metal container from rusting over time   # → route 'semantic', dense only
```

→ Next: **[Example 06](../example_06)** — Act B: read the passages into an answer.
