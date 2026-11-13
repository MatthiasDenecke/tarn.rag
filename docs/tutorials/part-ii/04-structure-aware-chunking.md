# 4 · Structure-aware chunking + auto-merge

> **The first rung that changes the *representation*** — so it re-ingests, into its own store.
> **Config:** `examples/part_ii/example_04/config.yaml`

[Rung 03](03-cross-encoder-reranking.md) ended with a failure no retrieval knob could fix: the best
chunk was half a procedure. The fix has to happen at ingest time.

Three changes from rung 03:

```yaml
  database:
    document_url: sqlite:///./examples/part_ii/store/structured.db   # ← its own store
  ...
    - class_name: Chunk
      chunker: { class_name: structure_aware, max_chars: 450 }       # ← was recursive
  ...
    merger: { class_name: auto_merge }                               # ← added
```

This is the **only** rung in Part II that re-ingests. Everything up to here was a config swap over one
index; changing the chunker changes what is *in* the index, so it needs its own store.

## Flat windows vs a section tree

`recursive` chunking produces flat, overlapping windows cut at character counts. It knows nothing
about the document — a heading is just more text, and a section boundary is invisible. That is why a
procedure got sliced in half.

`structure_aware` splits on the document's **headings** instead, producing a tree:

- **leaf chunks** — the paragraphs of a section, and
- **section parents** — a chunk spanning a whole section, emitted wherever a section groups two or
  more children.

Both levels go into the index and both are directly searchable (there is no level filter on the
retrievers). So a query that asks about a whole section can now *match* a whole section, because one
exists.

The base store held ~19 chunks; the structured store holds **21**. Those extra chunks are the section
parents.

## The fragment becomes an answer

```text
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
```

| | rung 03 (recursive) | rung 04 (structure-aware) |
|---|---|---|
| top-1 result | ❌ a ~300-char **fragment** — the first step-group only | ✅ a ~700-char passage: the **whole "Startup procedure" section** |
| contains "loading sequence" | no | **yes** |

The top result is now the complete section, every step included. Nothing about *retrieval* changed —
the same hybrid + reranker pipeline is running. The right answer simply exists as a chunk now, so it
can win.

## Auto-merge: the second half

The `merger` handles a related problem. When several sibling leaves of one section all match a query,
they crowd the top-k with near-duplicate pieces of the same thing — and squeeze out every other
document.

**Auto-merge** collapses retrieved siblings into their parent:

```text
tarn> explain compressor startup and shutdown
```

You will see a new **`merged`** stage in the breakdown — it does not appear in rungs 01–03. In its `Δ`
column a section parent appears (marked `＋`) and the redundant leaves drop out. Five fragments of
`compressor-startup` become one consolidated section, and the `final` table now holds **five distinct
documents** in the top 6 instead of being dominated by one document's pieces.

So the two changes do different jobs, and it's worth keeping them straight: **structure-aware chunking
is what makes a whole section retrievable**; **auto-merge is what stops one section's leaves from
crowding out everything else.** The first fixes rung 03's fragment. The second protects your context
window from redundancy.

## The coupling you must respect

`max_chars: 450` is **coupled to the corpus**. Keep every paragraph under roughly that budget, or the
structure-aware chunker will fall back to character-splitting it mid-word — reintroducing exactly the
fragmentation you came here to fix. The longest paragraph in `corpus-2` is 400 characters, which is
why 450 works.

This is a real constraint of the approach, not a wart in this example: structure-aware chunking
respects the structure a document *has*. A document with no headings, or with 2000-character
paragraphs, gives it nothing to work with.

## Run it

```bash
python -m examples.part_ii.example_04.run
```

Or interactively — remember this rung has its **own** store, so build it first:

```bash
python -m tarnrag.console examples/part_ii/example_04/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> status
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
tarn> explain compressor startup and shutdown
```

`status` reports **21 chunks** against the base store's ~19 — the section parents.

## Next

**[5 · Query routing →](05-query-routing.md)** — Act A's capstone. Stop fusing retrievers and start
*choosing* between them, per query.

---

[← Part II](index.md)
