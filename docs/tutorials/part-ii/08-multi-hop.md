# 8 · Multi-hop reasoning

> **Some answers can't be retrieved — only *reached*.**
> **Config:** `examples/part_ii/example_08/config.yaml` · **Needs an LLM key** · **its own store**

[Rung 07](07-grounding-and-abstain.md) made the system honest about what it couldn't support. This one
is about a question it cannot *reach*, no matter how good retrieval gets.

Two deltas from rung 07:

```yaml
  database:
    document_url: sqlite:///./examples/part_ii/store/multihop.db      # ← its own store
  ...
    reasoner: { class_name: decomposition, max_subquestions: 4, top_k: 8 }   # ← was single_hop
```

## The bridge

The corpus answers *"which oil grade does the TX-200 require?"* — but only across two documents:

```
compressor-models.md :  "The TX-200 is ... built on a Cooper-Bessemer GMV frame"
lubrication-spec.md  :  "Cooper-Bessemer GMV frames require ... ISO VG 68"
```

Neither holds the whole chain. The word **`TX-200` never appears in the lubrication spec**, and
**`ISO VG 68` never appears in the models document**. The link between them is the *frame name* — a
fact you only have after the first hop.

## Why one hop cannot do it

Here is the part that took an experiment to get right, and it is the point of the rung.

This example's store also holds four **near-identical frame specs** — Ariel JGT needs ISO VG 100,
Waukesha VHP needs ISO VG 46, Superior MH needs ISO VG 150, Clark HRA needs ISO VG 32. So when you ask
about oil grade for the TX-200, retrieval returns:

```text
retrieval top-4:  compressor-models · lubrication-ariel · lubrication-waukesha · lubrication-superior
MISS gold phrase 'ISO VG 68' in the top 4
```

The **right** spec — the GMV one — is crowded out. Four documents that look exactly as relevant to the
question ("oil grade", "compressor frame", "ISO VG") outrank it, because nothing in the *question*
distinguishes them. Only the bridge fact does, and the bridge fact is in another document.

That MISS is produced without any LLM at all. It is the crux: **no reasoner can read a passage that
retrieval never returns.** The failure is one of *reachability*, not of reading.

So `single_hop` — one retrieve, one read — answers honestly:

```text
[single hop (Example 07's reasoner)]  answer: not specified
MISS gold phrase 'ISO VG 68' in the answer
```

It didn't hallucinate a grade (it could have; three wrong ones were sitting in its context). It said
the truth: from what it was shown, the answer is not there.

Notice too that this non-answer is still reported **grounded**, and rung 07's abstain policy does
**not** fire — a non-answer makes no claims, so there is nothing for a grounding check to refute. The
verification layer is not a substitute for being able to *find* the evidence.

## Decomposition

```text
[decomposition]  answer: ISO VG 68     (grounded)
   cites: compressor-models · lubrication-spec
HIT  gold phrase 'ISO VG 68' in the answer
```

`decomposition` splits the question into sub-questions, retrieves for **each one**, and synthesizes the
gathered evidence. The second sub-question — about the frame, not about the TX-200 — retrieves the GMV
spec that the original question could never surface. Both bridge documents end up cited.

The library also ships **`iterative`**, a retrieve↔read loop where each hop can build on what the last
one found. It solves this question too (I checked). It is the better fit for *dependent* chains in
general, since it can use hop 1's answer to phrase hop 2; `decomposition` assumes the sub-questions are
independent and is cheaper. On this bridge, both land it — so the config pins the cheaper one.

## What it costs

The banner on this rung does not say "fails: …", because nothing about the *answer* regresses. The cost
is somewhere else, and it is real:

**`single_hop` pays one LLM call. `decomposition` pays several** — one to decompose, one read per
sub-question, one to synthesize — plus a retrieval round per sub-question. On a corpus where one hop
suffices (which was true for *every* question in rungs 06 and 07), that is pure overhead for an
identical answer.

This is the same shape as [rung 05](05-query-routing.md)'s lesson: a more powerful component is not a
free upgrade. Multi-hop is what you reach for when your questions genuinely bridge documents — not a
default to leave switched on.

## A note on the corpus

This rung ingests `corpus-2` **plus** `corpus-2-frames` (the four distractor specs) into its **own**
store, leaving the base store the other rungs share exactly as it was.

The distractors are not padding — they are what *makes the bridge a bridge*. Without them,
`lubrication-spec` is the only lubricant document in the corpus, so any question mentioning oil
retrieves it, both facts land in the context window together, and `single_hop` chains them perfectly
well. I measured that first: on the plain 12-document corpus, `single_hop`, `decomposition`, and
`iterative` **all** answer `ISO VG 68` correctly, and multi-hop demonstrates nothing.

That is worth stating plainly, because it is the real lesson about multi-hop: **it only pays when the
bridge document is genuinely out of reach.** On a small corpus where everything relevant fits in the
top-k anyway, one hop is enough — and you should not pay for hops you don't need. The four distractors
manufacture, in miniature, the condition that makes real corpora hard: many documents that look
equally relevant to the question, only one of which is right.

## Run it

```bash
pip install '.[openai]'   # key in OPENAI_LLM_KEY, or a repo-root .env
python -m examples.part_ii.example_08.run
```

Or interactively — this rung has its **own** store, so build it first (two ingests):

```bash
python -m tarnrag.console examples/part_ii/example_08/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> ingest examples/docs/corpus-2-frames
tarn> status
tarn> explain which oil grade does the TX-200 require?
tarn> ask which oil grade does the TX-200 require?
```

`status` reports **16 documents · 24 chunks**. Run the `explain` before the `ask`: seeing the GMV spec
*absent* from the candidates is what makes the answer's arrival surprising.

## Next

**Example 09 · the answerability gate** *(coming)* — rung 07 refuses *after* generating and verifying;
the gate refuses *before* spending the read, when the query's exact-match cues (an identifier, a quoted
phrase) appear nowhere in the evidence. Cheap check first.

---

[← Part II](index.md)
