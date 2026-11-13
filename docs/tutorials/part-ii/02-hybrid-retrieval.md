# 2 · Hybrid retrieval

> **Fusion fixes the part number — and breaks the paraphrase.** Nothing is free.
> **Config:** `examples/part_ii/example_02/config.yaml` · **Offline** · reads the base store (no re-ingest)

One change from the base config: a **sparse (BM25)** retriever beside the dense one, fused by
**Reciprocal Rank Fusion**.

```yaml
  retrieval_pipeline:
    retrievers:
      - class_name: dense
      - class_name: sparse        # ← added
    fuser: { class_name: rrf }    # ← was identity
```

That is the whole delta:

```bash
diff examples/part_ii/example_00/config.yaml examples/part_ii/example_02/config.yaml
```

## The probes swap

| Probe | Query | Rung 01 (dense) | Rung 02 (hybrid) |
|-------|-------|-----------------|------------------|
| part number | `XQ-9920-A` | ❌ MISS (rank 5) | ✅ **HIT** (rank 1) — sparse's exact match, fused in |
| paraphrase | *"service a rotary fluid machine…"* | ✅ HIT (rank 3) | ❌ **MISS** (rank 6+) — sparse noise demotes the semantic answer |

They **swap**. That is the entire lesson, and it is worth sitting with, because the aggregate story
("hybrid is better") would have hidden it completely.

## Why the part number now hits

Run `explain XQ-9920-A` and look at the two per-retriever tables — you now get one for `dense` *and*
one for `sparse`:

- **`sparse`** ranks `pump-maintenance` **#1**. BM25 matches terms, and `XQ-9920-A` is a term.
- **`dense`** still has it down around 5. Nothing changed there; it still has no meaning to embed.

RRF fuses the two *rankings* — not their scores, which are incomparable (a cosine distance and a BM25
value share no scale). A document ranked #1 by either retriever gets a strong contribution, so
`pump-maintenance` lands at rank **1** in the `final` table.

Rung 01's failure is repaired.

## Why the paraphrase now misses

Now run `explain service a rotary fluid machine before powering it up`:

- **`dense`** ranks `pump-maintenance` **#3** — it still understands the paraphrase perfectly.
- **`sparse`** ranks it **#12**. There are no shared words, so BM25 has no signal at all — and
  instead it confidently boosts documents that happen to share surface words like "machine",
  "powering", "startup": `compressor-startup`, `valve-maintenance`, and friends.

RRF blends the two, so BM25's *wrong* ranking gets an equal vote. `pump-maintenance` slips to rank
**6**, out of the top 4.

This is the part people miss about hybrid retrieval. Fusion doesn't just add sparse's strengths — it
adds sparse's **opinions**, including the confidently wrong ones. When one retriever has no signal, it
does not politely abstain; it returns its best guess, and fusion counts that guess.

**Lexical noise demoted a semantically correct hit.** That is the cost, and RRF has no way to know
which retriever was out of its depth on this query.

## So is hybrid worse?

No — and the honest answer is more interesting than either slogan.

Hybrid is the library's shipped default, because on real corpora it was measured never to lose to
dense-only on any evaluated segment (`doc/phases.md`). What this engineered corpus shows is not that
hybrid is bad, but the *mechanism* by which fusion can cost you: it is a rank-level democracy between
retrievers that are not equally competent on every query.

Two different repairs follow from that, and Part II does both:

- **[Rung 03](03-cross-encoder-reranking.md)** re-scores the fused candidates with a model that reads
  query and passage *together*, overruling the fusion when it was wrong.
- **[Rung 05](05-query-routing.md)** stops fusing altogether and instead *picks* the right retriever
  per query.

## Run it

```bash
python -m examples.part_ii.example_02.run
```

Or interactively:

```bash
python -m tarnrag.console examples/part_ii/example_02/config.yaml
tarn> explain XQ-9920-A
tarn> explain service a rotary fluid machine before powering it up
```

(If you haven't built the store yet: `tarn> ingest examples/docs/corpus-2` — it's idempotent.)

Each `explain` now prints **two** per-retriever candidate tables, `dense` and `sparse`, before the
`fused` → `final` stages. That second table is the whole rung; watch how RRF combines the two
rankings, and note that on the paraphrase the sparse column lights up on entirely the wrong
documents.

## Next

**[3 · Cross-encoder reranking →](03-cross-encoder-reranking.md)** — recover the paraphrase *without*
giving back the part number.

---

[← Part II](index.md)
