# Part II examples — a failure-driven complexity ladder

A new example track (the quick-start lives in `part_i/`). Each step adds **one** config knob, shows a
case that **works** and one that **fails**, and the *next* step's config repairs the failure. Every
step runs its good/bad cases through **both** `retrieve` (with a score breakdown) and `ask` (answer +
proof tree).

## Locked decisions

- **Console is the hero.** Each example *is* one or more `Settings` configs runnable via
  `python -m tarnrag.console cfg.yaml`; a thin `run.py` narrates the good-vs-bad cases and prints the
  explained breakdown.
- **YAML configs, fully explicit, heavily commented.** No reliance on Python defaults — every config
  enumerates every stage and component parameter (incl. all four `components` keys, since
  `Settings._fill_default_components` would otherwise fill them). **Acceptance test:** if a Python
  default changes, the example configs still produce the same results.
- **`explain()` is a `TarnRag` method that returns data, never renders.** It returns
  `Outcome[SearchExplanation]` (per-retriever candidates pre-fusion, fused order, pre/post-rerank
  movement, component scores, provenance). Rendering lives in the UI (console + the example runner) —
  preserving the documented output-free facade that the console and the tiqtasq REST API both rely on.
- **Ingest once.** Retrieval/generation steps are pure config swaps over one `base` store; only
  chunking/enrichment/extractor changes re-ingest, and those steps say so.

## Part A — Infrastructure (build first)

- [ ] **A1 — YAML config loading.** `Settings.from_file(path)` dispatches on extension
  (`.json` / `.yaml` / `.yml`); `TarnRag.__init__` uses it; PyYAML added to the `console`/`dev`
  extras (lazy import + actionable error). *(this step)*
- [ ] **A2 — `TarnRag.explain()` + `SearchExplanation`.** Output-free facade method returning the
  retrieval trace + scores + provenance. Needs A4.
- [ ] **A3 — Console `explain` command + score columns** that render a `SearchExplanation` (UI layer).
- [ ] **A4 — Retrieval trace seam.** The `Searcher` records intermediate stages (per-retriever
  candidates, pre-rerank order) so `explain()` can return them. Additive; the hot `retrieve` path is
  untouched.
- [ ] **A5 — Example runner harness.** `examples/part_ii/_runner.py`: load a YAML config, run a
  good + bad query through `retrieve`/`ask`, render via the UI helpers, print a
  "what this shows / what fails / fixed next" banner.
- [ ] **A6 — Corpus-2.** A crafted multi-doc corpus engineered so each failure below is reproducible.
- [ ] **A7 — Tests + no-defaults check.** Model-gated smoke tests per example (good case wins; each fix
  flips its predecessor's bad case) + a check that every config fully specifies its pipeline.
- [ ] **A8 — `examples/part_ii/README.md`** — prereqs (`pip install -e '.[onnx,console]'`,
  `python scripts/fetch_model.py`) + the store-reuse map.

## Part B — Design principles

1. One knob per step (the config diff is the lesson).
2. Explicit over implicit (configs name every component; stable against default changes).
3. Show, don't assert (every claim backed by printed scores / provenance / proof).
4. Failure-driven (each step motivated by the prior step's concrete failure).
5. Every step is a 2×2 — good + bad, each through retrieve **and** ask.
6. Config is the artifact; Python only narrates.
7. Ingest once; re-ingest only when the *representation* changes (and say so).
8. Reuse the eval harness (`sweep` / `format_segmented`) for aggregate scoreboards.

### Store-reuse map

| Store        | Chunker (re-ingest reason)            | Reused by                     |
|--------------|---------------------------------------|-------------------------------|
| `base`       | `recursive` (flat, no parents)        | P2-01, 02, 03, 05, 06, 07, 08 |
| `structured` | `structure_aware` (leaf+parent tree)  | P2-04 (auto-merge)            |
| `enriched`   | `recursive` + `Enrich(acronyms)`      | P2-09 (compared vs `base`)    |

All three ingest the same `corpus-2` Markdown (always `.md` → `markdown` extractor); they differ only in
the `Chunk`/`Enrich` stages. Parents are directly searchable (no level filter), so the flat `recursive`
chunker keeps the simple examples clean and `structure_aware` is its own store. See `CORPUS.md`.

## Part C — The ladder

**Act A — Retrieval** (over `base` unless noted)

- **P2-00 · Ingest the shared corpus** *(setup)* — fully explicit ingest.yaml; backs Acts A & B.
- **P2-01 · Dense-only** — good: paraphrase query; bad: rare-token/acronym query. → fixed by 02.
- **P2-02 · Hybrid (dense + sparse + RRF)** — fixes 01; bad: near-duplicate distractor outranks. → 03.
- **P2-03 · Cross-encoder rerank** — fixes 02 (rank-movement diff); bad: a section query returns several
  fragmented sibling leaves. → 04.
- **P2-04 · Auto-merge** *(re-ingest → structured / structure_aware)* — consolidates sibling leaves into
  the coherent section parent; bad: lexical query under-served. → 05.
- **P2-05 · Query routing** — fixes 04; capstone `sweep` scoreboard + classifier. Closes Act A.

**Act B — Generation** (over `base`)

- **P2-06 · Minimal generation (single_hop, no grounding)** — good: single-doc Q; bad: out-of-corpus
  hallucination. → fixed by 07.
- **P2-07 · Grounding check + abstain** — fixes 06; bad: multi-hop question partially grounded. → 08.
- **P2-08 · Multi-hop reasoning (decomposition)** — fixes 07; bad: acronym question misses. → 09.
- **P2-09 · Enrichment (acronyms)** *(re-ingest → enriched)* — fixes 08. Closes Act B.
- **P2-10 · Layout-aware extraction** *(optional capstone)* — extractor swap on a richer source doc.

## Build order

A1 → A6 → A2/A4 → A3 → A5 → P2-00 → P2-01 → P2-02 → … (one step per turn; each verified by
`python -m tarnrag.console …` + `python -m examples.part_ii.example_NN.run` + a test).
