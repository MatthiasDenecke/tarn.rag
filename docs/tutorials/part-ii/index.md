# Part II — the failure-driven ladder

[Part I](../part-i/index.md) teaches the machinery on a corpus small enough to check by eye. Part II
drives that machinery until it **breaks** — on purpose.

Every rung adds exactly **one** config knob and shows two queries: one that now works, and one that
**fails**. The next rung's config repairs that failure — and usually introduces a new one. The corpus
is engineered so each failure is reproducible rather than anecdotal.

The point is not that any single config is correct. It is that **every retrieval decision is a
trade**, and the only way to know what you traded away is to look at the query it broke.

## The ladder

| # | Adds | Fixes | Breaks |
|---|------|-------|--------|
| [00](00-ingest-the-corpus.md) | The base store | — | *(setup: the index every later rung reads)* |
| [01](01-dense-only-retrieval.md) | Dense-only retrieval | A paraphrase with no shared words | An opaque part number — nothing to embed |
| [02](02-hybrid-retrieval.md) | Sparse (BM25) + RRF fusion | The part number | The paraphrase — lexical noise demotes it |
| [03](03-cross-encoder-reranking.md) | A cross-encoder reranker | The paraphrase, *keeping* the part number | Fragmented answers — the best chunk is half a section |
| [04](04-structure-aware-chunking.md) | Structure-aware chunking + auto-merge | The fragment — whole sections come back | — |
| [05](05-query-routing.md) | Query routing *(Act A capstone)* | Per-query-type method selection: 1.00 on both types | — |
| [06](06-minimal-generation.md) | Generation *(Act B opener)* | Passages → an answer with a proof tree | It can't tell "I answered" from "I couldn't" |
| [07](07-grounding-and-abstain.md) | Grounding check + abstain | Unsupported answers — refuse instead | — |
| [08](08-multi-hop.md) | Multi-hop (`decomposition`) | Answers that bridge two documents — unreachable in one hop | — (but pays several LLM calls where one hop paid one) |

Rungs **01–05** are Act A (retrieval); **06–08** are Act B (generation). Walk them in order — the
failures only make sense in sequence.

**A ladder is not a stack.** Each rung repairs its predecessor's failure, but that does not mean you
should switch every knob on at once. Rungs 03 and 05 in particular turn out to be *alternatives*: the
library's measured sweep found the cross-encoder subsumes routing's gain entirely, so routing under a
reranker is pure cost ([rung 05](05-query-routing.md) has the numbers). Read the ladder as a tour of
the trade space, not a build order.

## Setup

```bash
pip install -e ".[onnx]"       # the package + the ONNX embedding runtime
python scripts/fetch_model.py  # the embedding model (once)
```

[Rung 03](03-cross-encoder-reranking.md) additionally needs the cross-encoder, and
[06](06-minimal-generation.md)–[07](07-grounding-and-abstain.md) need an LLM key — each page says so.

## How Part II is built

Three conventions, each chosen deliberately (`examples/part_ii/PLAN.md`):

**The console is the hero.** Each rung *is* a `Settings` config you can run interactively:

```bash
python -m tarnrag.console examples/part_ii/example_02/config.yaml
tarn> explain XQ-9920-A
```

`explain` is what makes the ladder legible: it prints the per-retriever candidates *before* fusion,
then the fused → reranked → merged → final stages with a movement (`Δ`) column. You watch the rank
move, and you see which component moved it. Each rung's `run.py` is a script that narrates the same
thing non-interactively.

**Configs are YAML, fully explicit, heavily commented.** Nothing relies on a library default — every
stage and component parameter is enumerated. The acceptance test is: *if a library default changes,
these examples still produce the same results.* This is not pedantry. While writing these pages the
library's default retrieval pipeline changed from dense-only to hybrid, and every Part II number was
unaffected, precisely because no config leaned on a default.

The corollary: **the diff between two configs is the lesson.**

```bash
diff examples/part_ii/example_01/config.yaml examples/part_ii/example_02/config.yaml
```

**Ingest once.** Rungs 01, 02, 03, 05, 06, and 07 are pure config swaps over the one store rung 00
builds. Only rung 04 changes the *representation* (how documents are chunked), so it re-ingests into
its own store — and it says so.

## The corpus

`examples/docs/corpus-2/` is 12 Markdown documents, each engineered to break something specific:

| Document | Engineered for |
|----------|----------------|
| `pump-maintenance.md` | Both directions at once: a **paraphrase** dense finds and BM25 misses, plus the opaque part number **`XQ-9920-A`** that dense misses and BM25 nails. |
| `pump-cavitation.md` | The true answer the reranker must recover. |
| `pump-vibration.md` | A **distractor** that shares "cavitation" and "noise" but answers nothing. |
| `compressor-startup.md` | A procedure split across paragraphs — the **fragmentation** case auto-merge repairs. |
| `compressor-models.md` + `lubrication-spec.md` | A **multi-hop bridge**: `TX-200` → `GMV frame` → `ISO VG 68`, which no single retrieve-and-read pass can gather. |
| `tank-corrosion.md` · `tank-inspection.md` · `valve-maintenance.md` · `ndt-methods.md` · `safety-ppe.md` | Domain depth, acronyms (`API 653`, `NDT`, `OSHA`, `LOTO`), and lexical near-pairs. |
| `quokka.md` | The off-topic distractor. Retrieval must discriminate; watch where it lands. |

They are Markdown **with headings** on purpose: structure-aware chunking needs a section tree to
build, and provenance needs a header path to cite. Design notes are in `examples/part_ii/CORPUS.md` —
deliberately *not* inside the corpus directory, since ingesting a directory ingests every file in it.

---

[← Tutorials](../index.md) · [← Part I](../part-i/index.md) · [Start the ladder →](00-ingest-the-corpus.md)
