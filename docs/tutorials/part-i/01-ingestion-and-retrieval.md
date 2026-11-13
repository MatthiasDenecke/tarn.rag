# 1 · Ingestion and retrieval

> **The smallest complete loop:** documents in, a queryable index out, ranked hits back.
> **Code:** `examples/part_i/example_01/{ingestion.py, retrieval.py}` · **Offline**

Every other tutorial in Part I is a variation on this one. Two scripts share one store: `ingestion.py`
turns three text files into an index, `retrieval.py` queries it.

## Run it

```bash
python -m examples.part_i.example_01.ingestion
python -m examples.part_i.example_01.retrieval
```

## Ingest

The whole example is one line of setup and one call:

```python
settings = base_settings(example_db(__file__))

async with TarnRag(settings) as tarn:
    outcome = await tarn.ingest([str(path) for path in DOC_PATHS])
```

`TarnRag` is the facade over the three engines. `async with` builds them — here an embedded SQLite
store and an in-process worker — and disposes the connection pool on the way out.

Because the mode is embedded, the whole pipeline runs in *this* process:

```
load + extract → enrich → clean → chunk → embed → persist
```

There is no queue to drain and no worker to wait on, so `ingest` returns only once every document is
written. Every status you get back is therefore **terminal** — which is exactly what makes embedded
mode good for learning and for tests.

The return value is an `Outcome`: `.value` holds the `DocumentStatus` list, and `.report` carries any
non-fatal issues — a path that didn't resolve, a document that failed. They are *reported*, never
silently dropped. The example prints both:

```text
Ingesting 3 documents from examples/docs/corpus-1 into examples/part_i/example_01/rag_docs.db

  pump-maintenance  complete   chunks=1  embeddings=1
  quokka            complete   chunks=1  embeddings=1
  tank-inspection   complete   chunks=1  embeddings=1

Stored 3 documents. Now query the same store:
  python -m examples.part_i.example_01.retrieval
```

Two things to notice.

**The ids are readable.** `pump-maintenance`, not a UUID. That is `ID_POLICY='caller'`: hand `ingest` a
file path and the file's stem becomes the document's stable id. Stable ids are what make a re-run an
*upsert* — run the script twice and you still have three documents, with chunks and embeddings
replaced rather than doubled.

**One chunk per document.** The default chunker (`structure_aware`) has a soft budget of 1200
characters, and these documents are about 200. So nothing splits. That is correct behaviour — don't
carve up a document that already fits — but it makes for a degenerate index: with one chunk per
document, "retrieval" is just document lookup, and every method scores perfectly. Tutorial 2 fixes
that, and tutorial 3 *needs* it fixed before comparing methods means anything.

## Retrieve

```python
settings = base_settings(db_path)          # same store, same embedding config

async with TarnRag(settings) as tarn:
    hits = (await tarn.retrieve(query, top_k=3)).value
```

`retrieve` embeds the query with the **same** embedder that embedded the passages, searches the
index, and returns up to `top_k` hits, best first. Building both sides from
`base_settings(example_db(__file__))` is what guarantees that shared identity — retrieval validates
the embedding fingerprint on open and refuses a mismatched store.

```text
Query: 'How do I check a storage tank for corrosion?'
  1. score=+0.033  document='tank-inspection'
     Storage tank inspection procedure: visually inspect the shell and roof for corrosion, me...
  2. score=+0.032  document='quokka'
     The quokka is a small macropod about the size of a domestic cat, native to Western Austr...
  3. score=+0.032  document='pump-maintenance'
     Centrifugal pump maintenance: check the mechanical seal for leaks, verify bearing lubric...

Query: 'Which animal is native to Western Australia?'
  1. score=+0.033  document='quokka'
     The quokka is a small macropod about the size of a domestic cat, native to Western Austr...
  2. score=+0.032  document='tank-inspection'
     Storage tank inspection procedure: visually inspect the shell and roof for corrosion, me...
  3. score=+0.016  document='pump-maintenance'
     Centrifugal pump maintenance: check the mechanical seal for leaks, verify bearing lubric...
```

Both queries rank the right document first. The second is a genuine *semantic* match: the quokka
document never contains the word "animal", and shares almost nothing with the query but "Western
Australia" — dense retrieval finds it on meaning. That is the thing keyword search cannot do, and
tutorial 3 measures how much it is worth.

Also note the quokka sitting at **rank 2** for the corrosion query. With three documents and a
`top_k` of 3, everything comes back no matter what — dense retrieval returns its nearest neighbours
and has no concept of "nothing here is relevant". Absolute rank is meaningful; mere presence is not.

## Reading the score

You configured no retrieval pipeline, so you got the default: **hybrid** — a dense retriever and a
sparse BM25 retriever, fused by **Reciprocal Rank Fusion**. RRF scores a chunk by where each
retriever *ranked* it, not by either retriever's raw score:

```
score(chunk) = Σ_retrievers  1 / (k + rank)          with k = 60 by default
```

Rank-based fusion is what lets you combine retrievers whose raw numbers are incomparable — a cosine
distance and a BM25 score have no common scale, but "you ranked it 1st" and "you ranked it 4th" do.

The printed numbers decode exactly:

| Score | What happened |
|-------|---------------|
| `+0.033` | **Both** retrievers ranked it #1 — `1/61 + 1/61 = 0.0328`. Dense and BM25 agree. |
| `+0.016` | Only **one** retriever returned it at all — `1/61 = 0.0164`. |

That `+0.016` on the last hit is informative: BM25 never returned `pump-maintenance` for "Which
animal is native to Western Australia?", because it shares no terms with it. Only dense did. The
score is telling you the retrievers *disagreed*, and you can see which one carried the hit.

The per-retriever numbers are not lost either — each hit keeps them in `component_scores`, so the
breakdown stays inspectable.

Two consequences worth carrying forward:

- **The magnitudes are small and clustered.** RRF scores live in a narrow band (`0.016`–`0.033` here)
  and are *not* probabilities or similarities. Never threshold on them ("only accept hits above
  0.5"); compare them only against each other, within one query.
- **Change the fuser and the score changes meaning entirely.** Configure a single dense retriever
  with the `identity` fuser and hits score `-rank` (`-1.0`, `-2.0`, `-3.0`) — higher is still better,
  but the magnitude carries no information at all. The score is an artifact of the pipeline you
  configured, not a property of the match.

To see that yourself, see [Swap the retrieval pipeline](../../how-to/swap-the-retrieval-pipeline.md).

## Try it

- Add a file to `examples/docs/corpus-1/` and re-run the ingestion. Only the new document is added;
  the rest upsert unchanged.
- Ask something the corpus cannot answer — `"What is the capital of France?"`. You still get three
  ranked hits. Retrieval always returns its nearest neighbours; deciding that *none of them support
  an answer* is a generation-side job, and it is exactly what tutorial 4 is about.

## Next

**[2 · A pipeline from JSON →](02-pipeline-from-json.md)** — the same engine, composed from a file
instead of the built-in default.

---

[← Part I](index.md)
