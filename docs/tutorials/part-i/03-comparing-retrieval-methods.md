# 3 · Comparing retrieval methods

> **Comparing methods is swapping a spec** — and scoring the result on a labeled set.
> **Code:** `examples/part_i/example_03/{evalset.json, ingestion.py, evaluation.py}` · **Offline**

Tutorial 2 made a pipeline into data. This is the payoff: if a retrieval method is just a spec, then
*comparing* methods is just a loop over specs — against one index, scored by one harness.

## Run it

```bash
python -m examples.part_i.example_03.ingestion    # small-chunk index (chunk_size 80)
python -m examples.part_i.example_03.evaluation   # sweep four pipelines over a labeled set
```

The ingestion is tutorial 2's small-chunk pipeline, and for the same reason: with one chunk per
document every method scores perfectly and the comparison is vacuous. Methods need room to disagree.

## Four pipelines, one index

```python
PIPELINES = {
    "dense":         {"class_name": "retrieval_pipeline",
                      "retrievers": [{"class_name": "dense"}]},
    "sparse (bm25)": {"class_name": "retrieval_pipeline",
                      "retrievers": [{"class_name": "sparse"}],
                      "fuser": {"class_name": "identity"}},
    "hybrid (rrf)":  {"class_name": "retrieval_pipeline",
                      "retrievers": [{"class_name": "dense"}, {"class_name": "sparse"}],
                      "fuser": {"class_name": "rrf"}},
}
```

Plus a fourth that *routes*: classify the query, then dispatch to whichever method is best for that
kind of query.

```python
ROUTED = {
    "class_name": "routing_retrieval_pipeline",
    "routes": {"lexical": PIPELINES["sparse (bm25)"], "semantic": PIPELINES["dense"]},
    "default": PIPELINES["dense"],
}
```

The sweep runs them all against the **same store**:

```python
async with TarnRag(settings) as tarn:
    ctx = tarn.retrieval_context()      # the shared (store + embedder) every spec runs against
    reports = await sweep({**PIPELINES, "routed (by type)": ROUTED}, ctx, evalset, k=3)
```

`retrieval_context()` is what makes this honest: every pipeline reads one index with one embedder, so
differences in the table are differences in *method*, not in data.

## The labeled set

`evalset.json` holds six queries, each tagged with a `query_type` and the gold phrases that make a
chunk relevant:

```json
{"text": "How do I keep a big metal container from rusting over time?",
 "relevant": ["corrosion"], "query_type": "semantic"}
{"text": "shell thickness ultrasonic testing",
 "relevant": ["ultrasonic", "shell thickness"], "query_type": "lexical"}
```

The split is the whole design. **Semantic** queries are paraphrases that deliberately share almost no
words with the documents ("big metal container" for a storage tank). **Lexical** queries are the exact
terms from the text. Relevance is content-based: a hit counts if the chunk text contains a gold phrase.

## The result

```text
pipeline                    hit@k      mrr   ndcg@k
---------------------------------------------------
dense                       1.000    0.722    0.804
sparse (bm25)               0.667    0.583    0.616
hybrid (rrf)                0.833    0.583    0.659
routed (by type)            1.000    0.722    0.804

(n=6 queries, k=3)
```

An aggregate table like this is where most evaluations stop, and it is where most of them go wrong.
It tells you dense won. It does not tell you *why*, or *when the winner would change*. Segment it:

```text
hit@k by type              lexical  semantic       all
------------------------------------------------------
dense                        1.000     1.000     1.000
sparse (bm25)                1.000     0.333     0.667
hybrid (rrf)                 1.000     0.667     0.833
routed (by type)             1.000     1.000     1.000
```

Now the mechanism is visible. **Sparse is perfect on lexical queries and collapses on semantic ones**
(0.333) — exactly as it must, because BM25 matches terms, and a paraphrase shares no terms. Dense
holds up on both. Hybrid lands in between: fusing sparse's rankings drags the semantic queries down
from dense's 1.000 to 0.667.

That is the question worth asking of any retrieval evaluation: **does a different method win on a
different kind of query?** If yes, a single pipeline is leaving quality on the table, and routing is
worth building. If no, routing is complexity for nothing.

## Two honest caveats

**Hybrid loses here, and it is still the shipped default.** Do not conclude from this table that
hybrid is bad. Three documents and six queries is not evidence about anything — on real corpora,
hybrid was measured never to lose to dense-only on any evaluated segment (`doc/phases.md`), which is
why it is the default you get when you configure nothing. What this toy corpus *does* faithfully show
is the failure *mode*: fusion is not free, and lexical noise can demote a semantically correct hit.
[Part II](../part-ii/index.md) builds an entire ladder on exactly that tension.

**Routing ties rather than wins.** `routed` matches `dense` to three decimals — because routing's
ceiling is the best method *on each type*, and here dense is already best on both. Routing only pays
when the per-type winners actually differ. (The harness tests contain the case where each method wins
one type and routing beats both; see `tests/eval/test_harness.py`.)

Both of these are more instructive than a table where the fancy method wins.

## Oracle routing, then the real classifier

The eval supplies each query's labeled `query_type`, and a caller-supplied type wins over the
router's classifier. So the routed pipeline above dispatches on the **labels** — it is an *oracle*.
That is on purpose: it isolates the question "does routing help?" from the separate question "can we
classify accurately?" If oracle routing doesn't beat a single pipeline, no classifier can rescue it.

Then the example checks the second question separately, with the domain-independent
`StructuralQueryClassifier` — no labels, no training, just query structure:

```text
StructuralQueryClassifier (domain-independent, no labels) vs the labels:
  [ok  ] predicted semantic  labeled semantic  How do I keep a big metal container from rusting over time?
  [ok  ] predicted semantic  labeled semantic  Which marsupial is famous for looking like it is grinning?
  [ok  ] predicted semantic  labeled semantic  How should I service a rotary fluid machine before powering it up?
  [ok  ] predicted lexical   labeled lexical   shell thickness ultrasonic testing
  [ok  ] predicted lexical   labeled lexical   quokka Western Australia macropod
  [ok  ] predicted lexical   labeled lexical   mechanical seal bearing lubrication shaft alignment
```

Six for six. The signal it keys on is structural, not semantic — questions with function words and
grammar are semantic; bare keyword strings are lexical — which is why it needs no domain knowledge.
So the routing the oracle simulated is the routing you would actually get in production.

## An aside worth noticing

These numbers are **byte-identical** to what this example produced before hybrid became the default
retrieval pipeline. Nothing here inherited a default: all four specs are stated in full, so a change
to the library's defaults cannot silently move the results. That property is what makes an evaluation
reproducible months later, and it is the convention every Part II config follows deliberately.

## Try it

- Add a query to `evalset.json` whose gold phrase appears in two documents, and watch nDCG separate
  the methods that rank *both* highly from those that find only one.
- Add a `hybrid (rrf)` route for `lexical` and re-run. Routing's ceiling changes with its routes.
- Weight the fusion — `{"class_name": "rrf", "weights": {"sparse": 2}}` — and see whether tilting
  toward BM25 recovers the lexical queries without costing the semantic ones.

## Next

**[4 · Grounded generation →](04-grounded-generation.md)** — retrieval finds passages. Now turn them
into an answer you can check, or a refusal you can trust.

---

[← Part I](index.md)
