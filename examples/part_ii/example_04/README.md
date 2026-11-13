# Example 04 · Structure-aware chunking + auto-merge

The **only** step that changes the representation, so it re-ingests into its **own** store
(`structured.db`). Three deltas from Example 03:

```yaml
  database:
    document_url: sqlite:///./examples/part_ii/store/structured.db   # ← separate store
  ...
    - class_name: Chunk
      chunker: { class_name: structure_aware, max_chars: 450 }       # ← was recursive
  ...
    merger: { class_name: auto_merge }                               # ← added
```

`structure_aware` splits on headings into **leaf chunks + section parents** (21 chunks vs the base
store's ~19), so Example 03's fragment becomes the **whole section** — every step, including the
loading sequence. **Auto-merge** then collapses retrieved sibling leaves into their parent, so one
section's pieces stop crowding the top-k.

⚠️ `max_chars: 450` is coupled to the corpus — keep paragraphs under it or the chunker character-splits
mid-word (corpus-2's longest is 400).

📖 **[Tutorial: Structure-aware chunking](../../../docs/tutorials/part-ii/04-structure-aware-chunking.md)**
— flat windows vs a section tree, and the two distinct jobs these changes do.

## Run

```bash
python -m examples.part_ii.example_04.run
```

...or interactively — this example has its **own** store, so build it first:

```bash
python -m tarnrag.console examples/part_ii/example_04/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> status
tarn> explain what are all the steps in the reciprocating compressor startup procedure?
tarn> explain compressor startup and shutdown     # watch the new `merged` stage
```

→ Next: **[Example 05](../example_05)** — query routing closes Act A.
