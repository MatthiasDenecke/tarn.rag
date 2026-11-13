# 2 · A pipeline from JSON

> **A pipeline is data.** Change the composition without touching Python.
> **Code:** `examples/part_i/example_02/{pipeline.json, ingestion.py, retrieval.py}` · **Offline**

Same engine as [tutorial 1](01-ingestion-and-retrieval.md), same corpus, same store layout. One
difference: the ingestion pipeline is read from a file.

## Run it

```bash
python -m examples.part_i.example_02.ingestion
python -m examples.part_i.example_02.retrieval
```

## The pipeline as a spec

`pipeline.json` is the entire lesson:

```json
{
  "class_name": "pipeline",
  "stages": [
    {"class_name": "LoadAndParse"},
    {"class_name": "Enrich", "enrichers": [{"class_name": "acronyms"}]},
    {"class_name": "CleanAndNormalize"},
    {"class_name": "Chunk", "chunker": {"class_name": "recursive", "chunk_size": 80, "overlap": 16}},
    {"class_name": "Embed"}
  ]
}
```

Every stage, extractor, enricher, and chunker in tarn.rag is a **Component**: it declares a
`class_name` and a config, and it is registered so it can be built from a plain dict. A pipeline is a
list of such specs — so the pipeline is *data*. Nothing above is a Python import. (The
[component catalog](../../reference/components.md) lists every tag and the slot it plugs into.)

Handing that spec to the engine takes one line:

```python
PIPELINE_SPEC = json.loads((Path(__file__).resolve().parent / "pipeline.json").read_text())

settings = base_settings(db_path, components={INGESTION_PIPELINE: PIPELINE_SPEC})
```

`Settings.components[INGESTION_PIPELINE]` is where the engine looks for the ingestion composition. If
you leave it unset, a validator fills in the default — which is what tutorial 1 got. Set it, and the
engine builds *yours*. Edit the JSON, re-run, no Python changes.

Two deltas from the default:

- **`Enrich` now has an `acronyms` enricher.** In the default pipeline `Enrich` runs with no enrichers
  at all — it is present as a seam, not as behaviour.
- **`Chunk` swaps `structure_aware` for `recursive` with `chunk_size: 80`.** This is the one that
  shows up in the output.

## What small chunks do

```text
Ingesting 3 documents with the pipeline from pipeline.json into examples/part_i/example_02/rag_docs.db

  pump-maintenance  complete   chunks=3  embeddings=3
  quokka            complete   chunks=3  embeddings=3
  tank-inspection   complete   chunks=4  embeddings=4

Stored 3 documents in 10 chunks (small-chunk pipeline; example 01's default made one per doc).
```

Ten chunks where tutorial 1 produced three. The corpus didn't change — the chunker did. Tutorial 1's
`structure_aware` chunker left these ~200-character documents whole (its budget is 1200);
`recursive` with `chunk_size: 80` cuts them into overlapping windows.

This matters more than it looks, because **chunking decides what a hit even is.** One chunk per
document makes retrieval a document lookup. Several chunks per document makes it *passage* retrieval:
hits get sharper, but also more fragmentary — and a single document can now occupy several of your
`top_k` slots. Retrieval shows exactly that:

```text
Query: 'How do I check a storage tank for corrosion?'
  1. score=+0.033  document='tank-inspection'
     Storage tank inspection procedure: visually inspect the shell and roof for ...
  2. score=+0.032  document='tank-inspection'
     ll and roof for corrosion, measure .  shell thickness by ultrasonic testing, ...
  3. score=+0.031  document='tank-inspection'
     re returning .  the tank to service.    ...
```

All three hits are now the **same document**. Whether that is an improvement depends entirely on what
happens next: it is excellent if you are feeding a reader model tight, on-topic passages, and it is a
waste of the context window if you wanted coverage across documents.

Notice too that `chunk_size: 80` is aggressive enough to split mid-word — `"ll and roof for
corrosion"` is the tail of "shell". The 16-character `overlap` is what keeps a severed phrase
recoverable from its neighbour. Small chunks buy precision and pay for it in coherence; there is no
universally correct setting, which is precisely why it belongs in a config file rather than in code.

## What the JSON deliberately does not set

The `Embed` stage carries no model configuration. That is not an oversight.

The embedding identity — which model, which dimension, which pooling — stays in `Settings.embedding`,
shared with retrieval, and the engine injects it into the `Embed` stage when it builds the pipeline.
If the JSON could name a model, a hand-edited pipeline could silently drift from the retrieval-side
embedder, and you would be embedding passages with one model and queries with another. Retrieval
would refuse to open the store (it checks the fingerprint), which is the *good* failure — but the
better design is to make the mistake unexpressible in the first place.

**Structure in JSON; embedding identity in Settings.**

## Retrieval doesn't need any of this

`retrieval.py` is tutorial 1's retrieval script, unchanged:

```python
settings = base_settings(db_path)   # no pipeline JSON anywhere
```

It never loads `pipeline.json`. Retrieval embeds a query and searches; it depends on the *embedding
identity*, not on how ingestion chunked. Chunking is baked into the index at ingest time, while the
embedder is a live contract between the two sides.

Internalise that asymmetry — it is the load-bearing fact behind Part II, where retrieval pipelines get
swapped freely over one store, but a chunking change forces a re-ingest.

## Try it

- Set `chunk_size` back to `512` and re-run the ingestion. Watch the chunk counts collapse toward one
  per document, and the three-hits-one-document effect disappear.
- Delete the `Enrich` stage. The pipeline still runs — stages are composable, not mandatory.
- Set `overlap` to `0` and look at the chunk boundaries. The severed phrases are now unrecoverable.

## Next

**[3 · Comparing retrieval methods →](03-comparing-retrieval-methods.md)** — now that a document is
more than one chunk, retrieval methods can actually disagree. Time to measure which one is right.

---

[← Part I](index.md)
