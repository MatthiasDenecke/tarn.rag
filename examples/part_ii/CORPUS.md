# corpus-2 — the engineered teaching corpus

`examples/docs/corpus-2/` is the substrate for the Part II ladder. The docs are **Markdown with
headings** (so structure-aware chunking can build leaf + section-parent trees for auto-merge, and so
provenance carries a header path) and deliberately **acronym-rich** (API, NDT, NPSH, OSHA, PPE, LOTO,
ISO VG, UT/RT/MT/PT) so the enrichment step has entities to annotate.

The directory holds **only content docs** — no README — because ingesting a directory ingests every
file in it. This design note lives here, one level up, on purpose.

## What each doc is engineered to demonstrate

| Doc | Engineered for | Key tokens / hook |
|-----|----------------|-------------------|
| `tank-inspection.md` | Lexical / exact-term content (`API 653`, `UT`) — but note `gte-small` handles these fine on this corpus, so they are **not** a dense failure | `API 653`, `UT`, inspection intervals |
| `tank-corrosion.md` | **Paraphrase/semantic** win (dense good case) | "loses metal to corrosion" ↔ "rusting" |
| `pump-maintenance.md` | **P2-01** both directions: paraphrase → dense win (sparse misses), AND the opaque part number `XQ-9920-A` → **dense miss** (no meaning to embed; sparse exact-matches → P2-02 hybrid fixes) | "rotary fluid machine … power up"; `XQ-9920-A` |
| `pump-cavitation.md` | **Rerank target** — the true answer (P2-02 wrong → P2-03 rerank fixes) | "prevent cavitation", `NPSH` |
| `pump-vibration.md` | **Rerank distractor** — shares "cavitation"/"noise" but doesn't answer prevention | "cavitation also produces … noise" |
| `compressor-startup.md` | **Auto-merge** — in the structure-aware store, the `## Startup procedure` section's 2 paragraphs become ≥2 sibling leaves under a section parent; a section-matching query returns several fragments that auto-merge consolidates into the parent (P2-03 fragmented → P2-04 merge fixes) | startup steps across two paragraphs |
| `compressor-models.md` | **Multi-hop bridge A** (P2-07 single-hop misses → P2-08 decomposition fixes) | `TX-200` → `Cooper-Bessemer GMV frame` |
| `lubrication-spec.md` | **Multi-hop bridge B** | `GMV frame` → `ISO VG 68` |
| `valve-maintenance.md` | Domain depth + a lexical near-pair (gate vs globe) | "gate valve stem packing" |
| `safety-ppe.md` | **Enrichment** showcase — dense with acronyms to annotate | `OSHA`, `PPE`, `LOTO` |
| `ndt-methods.md` | Acronym + semantic/lexical split; supports an acronym-recall story | "non-destructive testing (NDT)", `UT/RT/MT/PT` |
| `quokka.md` | Off-topic distractor (retrieval must discriminate) | — |

Out-of-corpus questions (e.g. "What is the capital of France?") have **no** supporting doc on purpose —
that is the P2-06 hallucination case that P2-07 grounding+abstain repairs.

## The `corpus-2-frames` add-on (P2-08 multi-hop)

`examples/docs/corpus-2-frames/` holds **four near-identical frame-lubrication specs** — Ariel JGT →
`ISO VG 100`, Waukesha VHP → `ISO VG 46`, Superior MH → `ISO VG 150`, Clark HRA → `ISO VG 32`. Only
**P2-08** ingests them (alongside `corpus-2`, into its own `multihop.db`), so the shared `base` store
and every rung that reads it are untouched.

They exist because **the multi-hop failure does not reproduce without them** (measured, 2026-07-13):
on plain `corpus-2`, `lubrication-spec` is the *only* lubricant document, so "which oil grade does the
TX-200 require?" retrieves it directly — both bridge facts land in the context window together and
`single_hop`, `decomposition` and `iterative` **all** answer `ISO VG 68`. Retrieval had already done
the "hop".

With the four distractors, the question's own vocabulary ("oil grade", "frame", "ISO VG") matches all
five specs equally, so the *right* one — the GMV spec — is crowded out of the top-k. It is now findable
only via the bridge entity (`TX-200` → *Cooper-Bessemer GMV frame*), which lives in another document.
`single_hop` answers "not specified"; `decomposition` bridges and answers correctly.

> The lesson generalizes: multi-hop only pays when the bridge document is genuinely **out of reach**.
> A corpus small enough that everything relevant fits in the top-k needs no hops — and shouldn't pay
> for them. The distractors manufacture, in miniature, the condition that makes real corpora hard.

## Three representations, one corpus

All three ingest the **same** `corpus-2` Markdown files (always routed `.md` → the `markdown`
extractor); they differ only in the `Chunk`/`Enrich` stages:

- **`base` store** — `recursive` chunker (flat overlapping windows, **no** parent tree): used by
  P2-01/02/03/05/06/07/08. Flat is the right default for the simplest examples — the `structure_aware`
  chunker indexes section parents *alongside* leaves (parents are directly searchable; there is no
  level filter on `dense_knn`/`sparse_search`), which would clutter a plain dense/hybrid demo with
  parent/leaf duplicates.
- **`structured` store** — `structure_aware` chunker at `max_chars ≈ 450` (every paragraph stays one
  clean leaf — the longest is 400 chars; multi-paragraph sections still split, e.g.
  `compressor-startup`'s 246+257-char startup paragraphs): used by **P2-04 (auto-merge)**. Because
  parents are also indexed, the P2-04 query must be chosen empirically so that **sibling leaves** (not
  the parent directly) dominate the top-k, giving auto-merge fragments to consolidate.
- **`enriched` store** — `recursive` + an `Enrich` stage (`acronyms`): used by **P2-09**, compared
  against `base`.

> ⚠️ `max_chars` is coupled to the corpus: keep every paragraph under ~450 chars or the
> `structure_aware` chunker will character-split it mid-word. Current longest paragraph: 400
> (`valve-maintenance.md`).

## Known refinement (P2-09 enrichment)

The built-in `acronyms` enricher only **tags** ALL-CAPS tokens as `entity` annotations (visible in
provenance) — it does **not** expand "non-destructive testing" → "NDT". So the P2-09 "acronym query now
matches" framing needs one of: (a) reframe P2-09 to show entity-annotation provenance, or (b) introduce a
small custom expansion `Enricher` (a nice "extend the system" example). The corpus already carries both
the expansion-with-acronym form (`ndt-methods.md`) and bare acronyms (`tank-inspection.md` uses `UT`), so
either path has material. Decide when we reach P2-09.
