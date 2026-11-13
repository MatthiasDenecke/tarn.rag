# 5 · Query routing

> **Act A's capstone.** Stop fusing retrievers. Start *choosing* between them, per query.
> **Config:** `examples/part_ii/example_05/config.yaml` · **Offline** · reads the base store

Every rung so far has hinted at the same thing: dense and sparse fail in *opposite* directions. Rung
01 showed dense nailing a paraphrase and missing a part number. Rung 02 showed sparse doing precisely
the reverse, and fusion splitting the difference — winning one probe and losing the other.

If a different method wins on a different **kind** of query, then the right move is not to blend them.
It is to *ask which kind of query this is* and dispatch accordingly.

```yaml
  retrieval_pipeline:
    class_name: routing_retrieval_pipeline
    classifier: { class_name: structural }
    routes:
      lexical:  { ... retrievers: [sparse] }   # exact terms / identifiers
      semantic: { ... retrievers: [dense]  }   # paraphrases
    default:    { ... dense + sparse + rrf }   # unknown → hybrid
```

## The evidence

Scored on a small labeled set (`evalset.yaml` — 4 semantic + 4 lexical queries), the split is stark:

```text
hit@k by type              lexical  semantic       all
------------------------------------------------------
dense                        0.750     1.000     0.875
sparse                       1.000     0.750     0.875
hybrid                       1.000     0.750     0.875
routed                       1.000     1.000     1.000
```

Look at the **`all`** column first, then never trust it again. All three fixed pipelines score an
identical `0.875`. On that aggregate they are indistinguishable — you would conclude the choice
doesn't matter.

The segmented columns say something completely different. **Dense owns the semantic queries** (1.000)
and drops on the lexical ones (0.750). **Sparse is exactly inverted** (1.000 / 0.750). They are not
three similar systems; they are two opposite systems and one blend, all averaging out to the same
number.

Routing's ceiling is the best method *on each type* — so it scores **1.000 on both**, beating every
fixed pipeline. The aggregate table was hiding a free win.

This is the single most transferable habit in Part II: **an aggregate metric can conceal the very
structure you need in order to improve.** Always segment by something you believe changes the answer.

## The classifier

Routing is only useful if you can tell the query types apart *at query time*, without labels.

The scoreboard above uses the labeled `query_type`, so the router dispatches on gold labels — **oracle
routing**. That is deliberate: it isolates "does routing help?" from "can we classify?" If oracle
routing didn't beat a fixed pipeline, no classifier could rescue it.

Then the second question gets answered separately. The **`StructuralQueryClassifier`** reproduces the
labels with none of its own — **7 of 8** on this set:

```text
  [ok ] predicted semantic  labeled semantic  service a rotary fluid machine before powering it up
  [ok ] predicted lexical   labeled lexical   XQ-9920-A
  [ok ] predicted lexical   labeled lexical   gate valve stem packing
  [MISS] predicted lexical  labeled semantic  find surface cracks without damaging the part
```

It keys on *structure*, not meaning: questions with grammar and function words are semantic; bare
keyword strings are lexical. That is why it needs no domain knowledge and no training — and also why
it misses the one it misses. "find surface cracks without damaging the part" is phrased like a
keyword string but wants a concept (non-destructive testing), and structure alone cannot see that.

A 7/8 classifier in front of a routing table that gains you 0.125 hit@k is still a clear win — but
note that the classifier's error rate now sits on the critical path, which is a cost the oracle table
does not show.

## Run it

```bash
python -m examples.part_ii.example_05.run
```

Prints the scoreboard, the classifier check, and two routed `explain` breakdowns. Or interactively:

```bash
python -m tarnrag.console examples/part_ii/example_05/config.yaml
tarn> explain XQ-9920-A
tarn> explain how do I keep a big metal container from rusting over time
```

Every query is now classified and routed, and the breakdown announces the decision on a `routed:` line
at the top:

- **`XQ-9920-A`** → `routed: query_type=lexical → route 'lexical'`. The only per-retriever table is
  **`sparse`** — the dense retriever is never run. `pump-maintenance` is rank 1.
- **"keep a big metal container from rusting"** → `routed: query_type=semantic → route 'semantic'`.
  The only table is **`dense`**, and `tank-corrosion` comes back on top, found purely by meaning.

Note what routing also buys you: it does *less work*. Only one retriever runs per query.

## Does routing survive contact with a real corpus?

Mostly not — and this is the most important paragraph on the page.

The library's own measured sweep (`doc/phases.md`, S1 — TAT-QA, n=334, source-hit@10) put routing
head-to-head with the profile from [rung 03](03-cross-encoder-reranking.md):

| pipeline | source-hit@10 |
|----------|---------------|
| hybrid + cross-encoder | **0.946** |
| routed (no reranker) | 0.937 |
| routed + cross-encoder | **0.946** — digit-identical to hybrid + CE |

**The cross-encoder subsumes the gain.** Routing and reranking are two ways of fixing the *same*
underlying problem — a retriever that ranked the right passage too low — and the cross-encoder is
simply better at it. Stack them and you pay for routing without getting anything: on that benchmark
`ROUTED+CE` and `HYBRID+CE` agree to three decimals.

So the naive reading of this ladder — "each rung adds a knob, so turn them all on" — is **wrong**, and
this rung is where it breaks. Rung 03 and rung 05 are alternatives, not layers.

Routing earns its place in a specific setting: **when you cannot afford a cross-encoder.** A reranker
costs a model forward pass per candidate per query, which is a real latency and hardware budget. If
that budget doesn't exist, routing recovers most of the gap LLM-free (0.937 vs 0.946) by protecting
the lexical/semantic split rather than correcting it after the fact.

That is the whole reason this rung's evaluation is worth running on *your* corpus rather than
inheriting a conclusion from anyone's table — including this one.

## That closes Act A

You can now retrieve well: dense for meaning, sparse for exact terms, a reranker to correct both, a
structure-aware index so whole sections are retrievable, and a router to pick the right method per
query.

Act B asks the next question — **what do you do with the passages once you have them?**

## Next

**[6 · Minimal generation →](06-minimal-generation.md)** — read the passages into an answer, and meet
the first failure of a system that trusts its reader.

---

[← Part II](index.md)
