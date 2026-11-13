# 1 · Dense-only retrieval

> **Dense embeddings match by meaning.** That is their strength, and their blind spot.
> **Config:** `examples/part_ii/example_00/config.yaml` (no delta yet) · **Offline**

The simplest retrieval there is: a single **dense** retriever — vector KNN over the embedded chunks,
with the `identity` fuser passing its ranking straight through. No config change from
[rung 00](00-ingest-the-corpus.md) yet; the first config *delta* is [rung 02](02-hybrid-retrieval.md).

## Two probes

Both queries target the same document, `pump-maintenance.md`. Dense retrieval gets one right and one
badly wrong.

| | Query | Dense |
|---|-------|-------|
| ✅ **good** | *"service a rotary fluid machine before powering it up"* — a paraphrase with **no words in common** with the document | **HIT at rank 3.** Found by meaning alone. BM25 ranks it **12th** — there is nothing to match on. |
| ❌ **bad** | `XQ-9920-A` — an opaque **part number** | **MISS.** The right document sits at rank **5**, outside the top 4. |

Read those two rows together, because they are the same fact seen from both sides.

The paraphrase works *because* the embedding captures meaning rather than surface form: "rotary fluid
machine" and "centrifugal pump" share no characters, and dense retrieval doesn't care. A keyword
search is helpless here.

The part number fails for exactly the same reason. `XQ-9920-A` **has no meaning to embed.** It is an
arbitrary string, and the embedding of the chunk containing it is "about pump maintenance" — not
about that token. So the vector for the query lands nowhere near it, and the correct document ranks
5th, below four documents that are vaguely about equipment.

An embedding model cannot represent a token whose entire value is its literal spelling. That is not a
tuning problem, and no amount of better embedding fixes it.

## Run it

```bash
python -m examples.part_ii.example_01.run
```

Or interactively, over the base config:

```bash
python -m tarnrag.console examples/part_ii/example_00/config.yaml
tarn> explain service a rotary fluid machine before powering it up
tarn> explain XQ-9920-A
```

## What `explain` shows you

The breakdown makes the *why* visible rather than asking you to take it on faith. For the paraphrase,
the dense retriever's candidate table surfaces `pump-maintenance` near the top. For the part number,
the same document is pushed down the list — and you can see the four irrelevant documents that
outrank it.

You will also notice the score column reads `-1.000`, `-2.000`, `-3.000`. That is the `identity`
fuser: with a single retriever there is nothing to fuse, so it preserves the retriever's ranking by
scoring each hit `-rank`. Higher is still better, but the magnitude carries no information. The raw
cosine distances are in the adjacent column (lower = nearer).

## The lesson

A single retriever has a *characteristic* failure, not a random one. Dense retrieval fails on
identifiers, codes, SKUs, error numbers, and names — precisely the queries where a user knows exactly
what they typed and expects an exact match.

The fix is not a better embedding. It is a **second retriever that works on a different principle**.

## Next

**[2 · Hybrid retrieval →](02-hybrid-retrieval.md)** — add a sparse BM25 retriever, the natural
complement to dense, and fuse the two. The part number starts hitting. Something else stops.

---

[← Part II](index.md)
