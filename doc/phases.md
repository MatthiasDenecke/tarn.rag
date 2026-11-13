# tarn.rag — performance roadmap toward SOTA

**Goal:** SOTA-competitive multi-hop QA (MOTHRAG ≈ 68.3 avg F1; NeocorRAG ≈ 69 is the GPU/constrained-decoding
ceiling) **without** trading away tarn.rag's differentiators (offline / layout-grade provenance / license
filtering). **Method: measure first, then add by measured leverage.** The MOTHRAG gap analysis lives in
`generation-architecture-design.md` §6; the eval harness is `scripts/run_benchmarks.py` (`tarnrag/eval/`).

## Roadmap, by leverage (cheap → expensive)

### Phase 0 — make the baseline trustworthy
Fix a target reader + embedder (either point at a real Llama-3.3-70B endpoint + a 3072-d embedder to match
MOTHRAG, or commit to our own default and own that number), run n ≥ 200/dataset, and sweep the reasoner
specs (`single_hop` / `iterative` / `decomposition`) — `sweep_benchmark` does this. This one step tells us
how much of the gap is **config** vs **architecture**.

### Phase 1 — cheap wins the eval flagged (config / prompt)
1. **Answer format / extraction** — the data shows `hit ≫ EM` (the reader finds the answer but returns prose,
   not a terse span; see the Phase 0 results below). A short-answer prompt + answer normalization is
   near-zero-cost and should move EM/F1 a lot. Do this first.
2. **Switch the default reasoner to `decomposition`** (multi-hop, already built; HotpotQA/MuSiQue are
   multi-hop). Config only.
3. **Stronger reader / embedder.** Config only.

Re-measure after each. Hypothesis: a meaningful chunk of the gap closes here for almost no code — and we'll
*know*, not guess.

### Phase 2 — architectural additions (§6), one at a time, each measured against Phase 1
1. **Ensemble + deterministic arbitration** (PDD, cross-arm agreement, faithfulness) — the single biggest
   MOTHRAG delta, and the heaviest.
2. **γ-driven re-retrieval** — couple the grounding checker back into the loop (today it's terminal).
3. **Bridge substrate** — multi-query fusion + an LLM relevance-judge reranker; targets recall (MuSiQue's
   known bottleneck).
4. **ChainFilter / OpenIE** — last; needs a triples/graph substrate; optional.

**Recommendation:** do Phase 0 + Phase 1 before building any Phase-2 component. They're cheap, they're exactly
what the architecture was built to make trivial (swap the spec), and they convert "behind by 0.27 F1" into
"X is config, Y is architecture" — which is what should decide whether the expensive ensemble/bridge work is
worth it. Starting with the ensemble would be optimizing the unmeasured.

---

## Phase 0 — results (2026-06-22)

### Setup
- **Harness:** `python scripts/run_benchmarks.py <dataset> --hf --limit 200 --sweep` (branch
  `feature/mothrag-phase0-sweep`, PR #71).
- **Protocol:** *distractor*, n = 200 per dataset. Each question ingests only its own candidate passages
  (~10 HotpotQA/2Wiki, ~20 MuSiQue) into an isolated store, then retrieve → reason → answer; scored
  token-F1 / exact-match (SQuAD-style, max over answer + aliases — the metric MOTHRAG reports).
- **Fixed:** reader `gpt-4o-mini` (OpenAI), embedder `gte-small` (local ONNX, 384-d), grounding **off**,
  `temperature=0`. **Swept:** the three reasoners.

### Results (n = 200 each; *hit* = a gold phrase appears in the answer, *cite* = gold support in cited passages)

**HotpotQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.585 | 0.456 | 0.265 | 0.696 |
| `iterative` | 0.601 | 0.184 | 0.005 | 0.799 |
| **`decomposition`** | **0.645** | **0.507** | **0.280** | 0.722 |

**2WikiMultiHopQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| **`single_hop`** | **0.654** | **0.414** | **0.170** | 0.739 |
| `iterative` | 0.467 | 0.130 | 0.000 | 0.857 |
| `decomposition` | 0.637 | 0.391 | 0.160 | 0.769 |

**MuSiQue**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.360 | 0.256 | 0.085 | 0.575 |
| `iterative` | 0.370 | 0.131 | 0.000 | 0.757 |
| **`decomposition`** | **0.445** | **0.275** | **0.075** | 0.655 |

**Best reasoner vs MOTHRAG (Llama-3.3-70B, n=1000)**
| dataset | best | F1 | F1\* | ΔF1 | EM | EM\* | ΔEM |
|---|---|---|---|---|---|---|---|
| HotpotQA | decomposition | 0.507 | 0.781 | −0.274 | 0.280 | 0.648 | −0.368 |
| 2Wiki | single_hop | 0.414 | 0.763 | −0.349 | 0.170 | 0.682 | −0.512 |
| MuSiQue | decomposition | 0.275 | 0.505 | −0.230 | 0.075 | 0.406 | −0.331 |
| **avg** | | **0.399** | **0.683** | **−0.284** | **0.175** | **0.579** | **−0.404** |

### Conclusions
1. **`decomposition` is the best default** — clearly best on HotpotQA + MuSiQue; on 2Wiki `single_hop` edges
   it (F1 0.414 vs 0.391, within noise). Best on average; the free config win holds.
2. **`iterative` format-collapses on all three** — F1 0.13–0.18, EM ≈ 0, *despite the best citation coverage*
   (0.76–0.86). It reasons and retrieves *well*, then narrates prose instead of a terse span. A systemic
   output-surface failure, not a reasoning one.
3. **`hit ≫ EM` everywhere → answer format is the dominant lever.** Starkest on 2Wiki (`single_hop` hit
   0.654 but EM **0.170** — a 0.48 gap: entity/date answers found but not emitted cleanly). A short-answer
   prompt + answer normalization attacks this directly, at near-zero cost.
4. **MuSiQue adds a second lever.** Its hit is far lower (0.445 vs ~0.65) — on 4-hop questions the content
   often isn't *found at all*, a genuine retrieval/multi-hop gap, not just formatting.

### Config-vs-architecture verdict
- **HotpotQA + 2Wiki are format-bound** — Phase 1 (decomposition default + short-answer/extraction) should
  close much of their gap with no architecture.
- **MuSiQue needs both** — format *and* retrieval/multi-hop capability. It's the dataset that justifies the
  Phase-2 bridge substrate / stronger multi-hop.
- Measure-first paid off: leading with the ensemble would have optimized the unmeasured, and the cheap
  levers (reasoner choice + answer surface) are the largest visible ones on 2 of 3 datasets.

### Methodology note
The first full sweep hit a transient DNS outage during the 2Wiki run (32 calls failed → the per-question
guard scored them empty and the run survived; the retry's ~15 s window was outlasted). 2Wiki was re-run
clean (0 failures) — and the clean numbers *flipped* the 2Wiki reasoner ranking vs the contaminated run, so
the re-run mattered. HotpotQA + MuSiQue were unaffected (0 failures).

### Caveats
n = 200/dataset (CI ≈ ±3.5 %); `gpt-4o-mini` reader (not MOTHRAG's Llama-3.3-70B); a single 384-d embedder
(`gte-small`). Absolute numbers will shift with a stronger reader, but the `hit ≫ EM/F1` **format** gap is
largely reader-independent.

### Status / next
- [x] All three datasets swept, clean (HotpotQA, 2Wiki, MuSiQue).
- [x] **Phase 1** done — see below.
- [ ] **Phase 2 candidate:** bridge substrate / stronger multi-hop — to push *past* MOTHRAG, esp. on MuSiQue.

---

## Phase 1 — results (2026-06-22)

**Changes:** default reasoner → `decomposition`; short-answer prompts (demand the minimal span, not a
sentence) + a `_clean_answer` lead-in/quote stripper. No architecture. Same harness/protocol as Phase 0
(`gpt-4o-mini` + `gte-small`, n=200, distractor, grounding off). Clean run, 0 failures.

### Results (n = 200 each)

**HotpotQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.536 | 0.659 | 0.550 | 0.681 |
| `iterative` | 0.432 | 0.523 | 0.430 | 0.767 |
| **`decomposition`** | **0.574** | **0.705** | **0.570** | 0.713 |

**2WikiMultiHopQA**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| **`single_hop`** | **0.648** | **0.651** | **0.575** | 0.704 |
| `iterative` | 0.418 | 0.393 | 0.335 | 0.814 |
| `decomposition` | 0.643 | 0.641 | 0.565 | 0.738 |

**MuSiQue**
| reasoner | hit | F1 | EM | cite |
|---|---|---|---|---|
| `single_hop` | 0.350 | 0.446 | 0.330 | 0.497 |
| `iterative` | 0.300 | 0.344 | 0.270 | 0.738 |
| **`decomposition`** | **0.415** | **0.513** | **0.390** | 0.530 |

### Phase 0 → Phase 1, best reasoner per dataset, vs MOTHRAG
| dataset | F1: P0 → P1 (Δ) | F1 gap to MOTHRAG | EM: P0 → P1 (Δ) | EM gap to MOTHRAG |
|---|---|---|---|---|
| HotpotQA | 0.507 → 0.705 (+0.198) | −0.076 | 0.280 → 0.570 (+0.290) | −0.078 |
| 2Wiki | 0.414 → 0.651 (+0.237) | −0.112 | 0.170 → 0.575 (+0.405) | −0.107 |
| MuSiQue | 0.275 → 0.513 (+0.238) | **+0.008** | 0.075 → 0.390 (+0.315) | −0.016 |
| **avg** | **0.399 → 0.623** | **−0.060** | **0.175 → 0.512** | **−0.067** |

### Conclusions
1. **The format fix was decisive.** The gap to MOTHRAG collapsed from **−0.28 F1 / −0.40 EM** (Phase 0) to
   **−0.06 F1 / −0.07 EM** — ~80 % closed with **zero architecture**, just the prompt + reasoner default.
   Emphatic validation of measure-first.
2. **MuSiQue reaches MOTHRAG parity** (F1 0.513 vs 0.505) — even though it was the "retrieval-bound" dataset.
   Answer format was the dominant lever *everywhere*, including the hard one. (Its hit is still ~0.42, so
   retrieval caps the ceiling — Phase 2 retrieval work would push it *past* MOTHRAG, not just to parity.)
3. **`iterative` recovered but still lags** — F1 0.18→0.52 / 0.13→0.39 / 0.13→0.34 (the short-answer prompt
   fixed its prose-narration collapse), yet it trails on hit: its follow-up loop gathers good evidence (best
   citation coverage) but finds the answer less often. Not the default.
4. **`decomposition` confirmed as the default** — best on HotpotQA + MuSiQue; `single_hop` still edges it on
   2Wiki, within noise.

Note: "hit" fell on some arms vs Phase 0 because Phase 0's verbose answers inflated it (a gold phrase appears
*somewhere* in a long answer); with terse answers, hit ≈ correctness, and F1/EM are the trustworthy metrics.

### Caveats
Same as Phase 0: n = 200 (CI ≈ ±3.5 %); `gpt-4o-mini`, not Llama-3.3-70B; 384-d `gte-small`. A controlled
match of MOTHRAG's reader/embedder would shift absolutes; the *relative* Phase-0→Phase-1 gain is robust.

### What's next
The cheap levers are nearly exhausted (~0.06 F1 off MOTHRAG). To go *past* it, **Phase 2** — and the data
says **retrieval is now the ceiling** (hit 0.42–0.65): the bridge substrate (multi-query fusion + an LLM
relevance-judge reranker) and stronger multi-hop, measured against this Phase-1 baseline. A stronger
reader/embedder (the other Phase-1 lever, not yet pulled) is a parallel free check — done next.

### Reader check — gpt-4o (2026-06-22)

The one Phase-1 lever not yet pulled: swap the reader to **gpt-4o** (everything else identical — `gte-small`,
n=200, same prompts, `--sweep`), to see how much of the residual gap is the reader. Clean run, 0 failures.

**gpt-4o results (n=200; decomposition is now best on all three)**
| dataset | reasoner | hit | F1 | EM |
|---|---|---|---|---|
| HotpotQA | `decomposition` | 0.639 | **0.766** | **0.625** |
| 2Wiki | `decomposition` | 0.747 | **0.726** | **0.650** |
| MuSiQue | `decomposition` | 0.540 | **0.617** | **0.505** |

**Best reasoner per dataset: Phase 1 (gpt-4o-mini) → gpt-4o, vs MOTHRAG**
| dataset | F1: mini → 4o | F1 gap to MOTHRAG | EM: mini → 4o | EM gap to MOTHRAG |
|---|---|---|---|---|
| HotpotQA | 0.705 → 0.766 | −0.015 | 0.570 → 0.625 | −0.023 |
| 2Wiki | 0.651 → 0.726 | −0.037 | 0.575 → 0.650 | −0.032 |
| MuSiQue | 0.513 → 0.617 | **+0.112** | 0.390 → 0.505 | **+0.099** |
| **avg** | **0.623 → 0.703** | **+0.020** | **0.512 → 0.593** | **+0.014** |

**Conclusions**
1. **The residual gap was the reader.** gpt-4o lifts best-reasoner average F1 0.623 → **0.703** and EM 0.512 →
   **0.593** — *at/above* MOTHRAG's average (0.683 / 0.579). HotpotQA + 2Wiki land within −0.02…−0.04;
   MuSiQue is well **above** (+0.11 F1).
2. **The headline:** tarn.rag's **lean** stack — no ensemble, no bridge, no ChainFilter — matches MOTHRAG's
   average on a comparable reader. The §6 architecture is **not needed for parity**; it's for going *past*.
3. **Caveat — not a controlled win.** gpt-4o is very likely a stronger reader than MOTHRAG's Llama-3.3-70B,
   so "above MOTHRAG's average" reflects a reader edge *and* the Phase-1 prompts, not architecture. The
   defensible claim: **the residual Phase-1 gap was dominated by reader strength, not by the missing
   architecture.** A true match needs tarn.rag *on* Llama-3.3-70B (an OpenAI-compatible endpoint).
4. **decomposition is now best on all three** (it overtook `single_hop` on 2Wiki with the stronger reader) —
   the default is firmly vindicated.
5. `hit` also rose with gpt-4o (0.54–0.75) — some of gpt-4o-mini's low hit was weak extraction, not just
   retrieval misses; but headroom to 1.0 remains.

This **reframes Phase 2**: not "catch up to MOTHRAG" (parity is reached) but "go beyond" — raise retrieval
`hit` toward its ceiling (bridge substrate), with MuSiQue the clearest target. A controlled run on a
Llama-3.3-70B endpoint would also pin the apples-to-apples number.

---

## Phase 2 — results: bridge substrate (2026-06-23)

**Feature:** the bridge retrieval substrate (PR #73) — `multi_query` (LLM query expansion → dense per
variant → RRF) + `llm_judge` (LLM relevance reranker). **Eval:** `--bridge --sweep`, same protocol as
Phase 1 (`gpt-4o-mini` + `gte-small`, n=200, distractor), so the per-reasoner table overlays Phase 1's.
Clean run, 0 failures.

### Bridge results (n = 200 each)
| dataset | reasoner | hit | F1 | EM |
|---|---|---|---|---|
| HotpotQA | `single_hop` | 0.590 | 0.716 | 0.585 |
| HotpotQA | `iterative` | 0.497 | 0.582 | 0.470 |
| HotpotQA | **`decomposition`** | 0.596 | **0.729** | 0.600 |
| 2Wiki | `single_hop` | 0.659 | 0.649 | 0.575 |
| 2Wiki | `iterative` | 0.291 | 0.295 | 0.245 |
| 2Wiki | **`decomposition`** | 0.687 | **0.676** | 0.585 |
| MuSiQue | `single_hop` | 0.410 | 0.486 | 0.375 |
| MuSiQue | `iterative` | 0.300 | 0.340 | 0.275 |
| MuSiQue | **`decomposition`** | 0.450 | **0.534** | 0.420 |

### Δ vs Phase 1 (baseline dense retrieval), `decomposition` (the default)
| dataset | hit | F1 | EM |
|---|---|---|---|
| HotpotQA | 0.574 → 0.596 (+0.022) | 0.705 → 0.729 (+0.024) | 0.570 → 0.600 (+0.030) |
| 2Wiki | 0.643 → 0.687 (+0.044) | 0.641 → 0.676 (+0.035) | 0.565 → 0.585 (+0.020) |
| MuSiQue | 0.445 → 0.450 (+0.005) | 0.513 → 0.534 (+0.021) | 0.390 → 0.420 (+0.030) |
| **avg** | | **0.620 → 0.646 (+0.026)** | **0.503 → 0.535 (+0.032)** |

### Conclusions
1. **The bridge helps — modestly and consistently.** `decomposition` gains +0.026 avg F1 / +0.032 avg EM
   (gpt-4o-mini); positive on every dataset. **MuSiQue `decomposition` now exceeds MOTHRAG F1** (0.534 vs
   0.505) *even on gpt-4o-mini*. It closes ~40 % of the residual gpt-4o-mini→MOTHRAG F1 gap (−0.060 → −0.037).
2. **`single_hop` benefits most** (+0.057 F1 HotpotQA, +0.040 MuSiQue) — it does *one* retrieval, so better
   ranking helps it directly; `decomposition` already multi-retrieves per sub-question, so it gains less.
3. **`iterative` + bridge regresses** — −0.098 F1 on 2Wiki (hit 0.418 → 0.291). Two query-reshaping
   mechanisms (the iterative follow-up loop + multi-query/judge) interact badly. Caution: don't pair them.
   (`iterative` isn't the default, so the headline is unaffected.)
4. **The gains are small because distractor bounds the bridge.** Each question's store holds only ~10–20
   passages, so the bridge can only re-rank that pool into the top-8 — it can't *recall* missing evidence.
   That's why MuSiQue's `hit` barely moved (+0.005): its 4-hop depth isn't a ranking problem.

### Verdict — fullwiki is the next gate
The bridge's real value — recall over a large corpus — is **not exercised in distractor**. The modest but
positive distractor gains confirm it works; the *order-of-magnitude* payoff needs the **fullwiki** setting
(retrieve over the whole Wikipedia corpus, not a per-question pool). So the next required step before more
retrieval architecture (γ-retrieval, ChainFilter) is **fullwiki ingestion + retrieval**, against which those
levers — and the bridge itself — can actually show their worth. Building more retrieval cleverness against a
20-passage pool would be optimizing the wrong setting.

---

## Phase 2.5 — shared-corpus (pool) investigation (2026-06-24)

Built the shared-corpus harness (one persistent index, retrieve-only per question; `--corpus pool`), plus
the supporting hardening: **bounded concurrency** for the eval (~7× — the LLM calls are I/O-bound), an
**`llm_judge` shortlist cap** (corpus retrieval hands the reranker 100+ candidates), the OpenAI retry
**honoring `Retry-After`**, and a per-embedder index cache key. Ran HotpotQA over a **~10K-passage pool**
(1,000 dev questions, deduped), decomposition, n=200.

### Results (pool ~10K, decomposition)
| variant | hit | F1 | EM |
|---|---|---|---|
| gte-small + gpt-4o-mini (baseline) | 0.508 | 0.633 | 0.515 |
| + bridge (multi-query + judge) | 0.497 | 0.632 | 0.515 |
| gte-small + **gpt-4o** | 0.541 | 0.653 | 0.540 |
| **text-embedding-3-small** (1536-d) + mini | 0.514 | 0.644 | 0.515 |
| *MOTHRAG* | — | *0.781* | *0.648* |

(Distractor reference: gte-small + gpt-4o-mini decomposition was hit 0.574 / F1 0.705.)

### Conclusions
1. **`hit` is stuck at ~0.51–0.54 across every lever** — bridge, reader, and a much stronger embedder all
   hit the same wall. So the recall ceiling is **not** embedder quality, the reader, or query expansion.
2. **The bridge is redundant with decomposition** (flat) — decomposition already expands the query.
3. **The reader is modest on the pool** (gpt-4o +0.02 F1 vs distractor's +0.06) — it can only answer what's
   retrieved, and retrieval misses ~half.
4. **A stronger embedder is flat** (te3-small ≈ gte-small) — so we did *not* pursue a local strong embedder.
5. The wall is the **multi-hop retrieval problem**: HotpotQA's 2nd-hop passage isn't similar to the
   *question* (it's similar to the 1st hop's *answer*), so dense retrieval can't surface it — no matter the
   embedder. The cheap levers (reader / embedder / bridge) are **spent** on the realistic pool setting.

### Verdict → Phase 3
tarn.rag is at MOTHRAG parity on **distractor** (lean stack + gpt-4o); the **pool** gap is the multi-hop
retrieval ceiling. The levers that actually attack it are the §6 items *not* redundant with decomposition:
**γ-driven re-retrieval** (a failed grounding check triggers a follow-up search using what's been read — it
can find the missing hop) and, heavier, ChainFilter. Phase 3 = **γ-driven re-retrieval**.

---

## Phase 3 — results: γ-driven re-retrieval (2026-06-24)

`GroundedRetrievalReasoner` (`grounded_retrieval`) — retrieve → read → grounding-check; an ungrounded claim
triggers a follow-up search *for that claim* (the bridge passage), then re-read. Eval: pool ~10K, gte-small,
gpt-4o-mini, n=200, **heuristic** grounding (the LLM-free default).

| reasoner | hit | F1 | EM |
|---|---|---|---|
| decomposition | 0.481 | 0.613 | 0.500 |
| **grounded_retrieval (γ, heuristic)** | **0.508** | **0.621** | **0.510** |

**Verdict: a small, partly-within-noise positive** (+0.027 hit; F1/EM within the n=200 ~±0.02–0.03 floor —
decomposition read 0.633 last run vs 0.613 here). γ-retrieval *nudges* the multi-hop ceiling but doesn't
break it with heuristic grounding. Open levers: (a) **LLM grounding** (sharper gap detection → better-targeted
re-retrieval — cheap config try), (b) **ChainFilter** (the heavy §6 item: OpenIE triples + chain density),
or (c) bank it — tarn.rag is at MOTHRAG parity on distractor; the realistic-pool multi-hop recall is a hard
frontier that incremental levers (bridge, embedder, γ-heuristic) only nudge.

### γ with LLM grounding — the bottleneck is re-retrieval, not the signal (2026-06-24)

Re-ran γ with `--grounding llm_grounding` (sharper claim-gap detection) vs decomposition, same protocol:

| reasoner | hit | F1 | EM |
|---|---|---|---|
| decomposition | 0.481 | 0.610 | 0.495 |
| grounded_retrieval (γ, **llm_grounding**) | 0.486 | 0.596 | 0.480 |

**LLM grounding made γ worse, not better** (−0.014 F1 vs decomposition; below γ-heuristic's 0.621 F1). So the
weak link is **not** the grounding signal — it's the re-retrieval itself: claim-as-query dense retrieval still
can't surface the question-dissimilar bridge passage, and the extra hops add context noise. **γ-retrieval is
spent** as a pool lever.

### Phase 3 verdict (final)
Every measured lever — bridge, stronger reader, stronger embedder, γ-heuristic, γ-llm — only *nudges* the
pool's hit ≈ 0.51 multi-hop recall ceiling. The multi-hop retrieval problem (the 2nd-hop passage isn't similar
to the question) is a **genuine hard frontier** that incremental retrieval tricks don't crack. The honest
state: **tarn.rag matches MOTHRAG on distractor with a lean, differentiated stack** (offline / layout
provenance / license filtering); the realistic-pool multi-hop recall gap is characterized and bounded. The
only untried lever is **ChainFilter** (OpenIE triples + chain density — heavy, uncertain payoff against a
ceiling 5 levers couldn't move).

---

## Post-MOTHRAG performance plan

Stepping back from the benchmark chase: "performance" is several axes, and the MOTHRAG setting (multi-hop
Wikipedia QA) doesn't exercise tarn.rag's differentiators (offline / layout provenance / licensing). Four
options, in order: (1) hybrid retrieval, (2) differentiators (attribution + layout, on real docs), (3)
systems (bulk-ingest throughput + latency), (4) robustness / eval on real data.

### Option 1 — hybrid retrieval (dense + BM25, RRF) (2026-06-24)

The whole pool push was dense-only. BM25 matches exact entity tokens (the bridge entity that embeds weakly
against the question), so hybrid targets the dense-only recall ceiling — at no LLM cost, and no rebuild (the
FTS5 index is built at ingest). Pool ~10K, decomposition, gpt-4o-mini, n=200, same session:

| retrieval | hit | F1 | EM |
|---|---|---|---|
| dense (baseline) | 0.497 | 0.623 | 0.505 |
| **hybrid (dense + BM25, RRF)** | **0.508** | **0.638** | **0.510** |

**+0.011 hit / +0.015 F1 / +0.005 EM** — small but consistent across all three, and **free** (no LLM, no
rebuild): the best cost/benefit lever found on the pool. It doesn't *break* the multi-hop ceiling (BM25 can't
surface a bridge entity absent from the question), but it's a worthwhile default. Enabled via `--hybrid`
(`HYBRID_RETRIEVAL`).

### Option 2 (PR-1) — layout-aware retrieval on TAT-QA (2026-06-24)

The MOTHRAG setting can't test tarn.rag's differentiators. TAT-QA (financial **table** + **paragraphs**,
with an `answer_from` ∈ table/text/table-text label) can: build one shared corpus of every table + paragraph,
then for each extractive question measure **source-hit@k** (did the top-k include the gold answer-source
element), segmented by where the answer lives. 100 records → 612 elements, 334 extractive queries, gte-small.

| segment | dense | hybrid | Δ |
|---|---|---|---|
| table | 0.779 | 0.853 | +0.074 |
| table-text | 0.855 | 0.945 | +0.090 |
| text | 0.938 | 0.953 | +0.015 |
| **overall** | 0.865 | 0.922 | +0.057 |

**Findings:** (1) **dense has a table deficit** — tables 0.779 vs text 0.938 (a 0.16 gap): cell tokens embed
weakly against a NL question. (2) **hybrid's real home is tables** — BM25 matches exact cell tokens, lifting
table/table-text far more than text (+0.074 / +0.090 vs +0.015); hybrid's +0.057 overall here dwarfs its
+0.015 F1 on Wikipedia QA. The differentiated setting reveals a lever the benchmark hid. Run via
`scripts/run_layout_eval.py`. Next (Option 2 PR-2): attribution precision (LLM-judge / `grounded_rate`) on
TAT-QA; later, ingest tables through the native structured path (Table elements) instead of rendered text.

### Option 2 (PR-2) — attribution precision on TAT-QA (2026-06-25)

Beyond *finding* the source (PR-1), does the answer *attribute* to it? Answer each extractive question with a
`single_hop` reader over hybrid retrieval, then have an LLM judge whether each cited span supports its claim
(`grounding_checker: llm_grounding`). `grounded_rate` = attribution precision; `citation_coverage` = is the
gold answer span present in the cited evidence. n=334, gte-small + gpt-4o-mini.

| segment | n | F1 | EM | attrib | cite |
|---|---|---|---|---|---|
| table | 95 | 0.509 | 0.379 | 0.884 | 0.732 |
| table-text | 110 | 0.571 | 0.427 | 0.936 | 0.706 |
| text | 129 | 0.595 | 0.349 | 0.992 | 0.884 |
| **overall** | 334 | 0.563 | 0.383 | 0.943 | 0.782 |

**Findings:** (1) **attribution precision is high (0.94 overall)** — when tarn.rag answers, the cited spans
support the claim 94% of the time. (2) **A consistent table penalty across every metric** — F1 0.51 vs 0.60,
attribution 0.88 vs 0.99, citation-coverage 0.73 vs 0.88 (table vs text). Tables are systematically harder to
answer *and* attribute, which points at the **linearized-table representation** (rendered to text) as the
next lever — the motivation for ingesting tables through the native structured `Table`-element path (a
follow-up). Run via `scripts/run_layout_eval.py --attribution`.

---

## Option 3 — bulk-ingest throughput (2026-06-25)

The per-doc ingest path gated every corpus build this session (~2/s). Profiling a 100-doc ingest (offline
hash embedder, isolating orchestration from embedding) found **~64 `engine.begin()` transactions per
document** — transaction-count *overhead* (~5 ms each of asyncio+SQLAlchemy machinery; not fsync, not the
inserts, which total ~1 s). Two causes:
1. the embedded queue dispatched **one document at a time** (`Batch([job])`), so each doc ran the DAG solo;
2. **two per-doc loops** persisted per document even within a batch — `DocumentResultSink._persist`
   (`store_document` per doc) and the orchestrator's `_record` (status per job, per stage).

**Fix (two parts):**
- **Batched dispatch** — `InMemoryJobQueue` groups a wave's queued jobs by `stage_name` into homogeneous
  `Batch`es (capped at `max_batch_size`), so a stage runs over many docs at once (the worker/sink already
  supported multi-job batches). On a batch failure it re-runs each job solo, preserving per-job
  at-least-once / dead-letter semantics.
- **Bulk-persist** the two hot loops — `store_documents` (one transaction for the batch's documents) and
  `record_jobs` (one transaction for the batch's status rows), reusing the exact per-row logic.

**Result** (300 docs, offline): **3.6 → 9.4 docs/s, ~2.6×**. Equivalence verified — the batched + bulk index
is identical to the per-doc index (same documents, chunks, embeddings); only the transaction count drops.
Batched dispatch also batches the **Embed** stage, so API embedders make far fewer requests (the cost that
made the te3 corpus build take ~2.6 h). Remaining floor: per-row inserts + ~5 stages; further wins would need
cross-stage batching.

---

## Option 4 — robustness: deterministic drift-guards (2026-06-25)

The gains this track measured (hybrid retrieval, attribution, ingest throughput) had no regression guard — a
refactor could silently undo them. CI shapes what a guard can check: the lean push CI (`codecov.yml`) has no
models / datasets / LLM key; the gated `model-tests.yml` job has the ONNX models but still no LLM key; and the
LLM metrics (attribution, multi-hop F1) are non-deterministic (gpt-4o-mini ±0.02–0.03). So the **deterministic,
every-push** guards are:

- **Hybrid mechanic** (`tests/retrieval/test_hybrid_regression.py`) — the *cause* of the TAT-QA table lift,
  distilled into an invariant: with a controlled embedder, a 'table' chunk whose exact tokens are present but
  whose dense vector is far from the query is ranked **below** the top-k by dense-only, and **rescued** by
  hybrid (BM25 → RRF ranks it first). Breaks if the sparse arm or RRF fusion regresses. No model needed.
- **Ingest equivalence** (`tests/ingestion/test_bulk_ingest.py`, from Option 3) — the batched + bulk-persist
  index is identical to the per-doc index.

The full, real-data quality metrics (gte-small table lift, gpt-4o-mini attribution / multi-hop F1) need
corpus scale and/or an LLM key, so they're tracked **on-demand** via `scripts/run_layout_eval.py` and
`scripts/run_benchmarks.py` rather than as committed CI assertions.

---

## P2 + P3 — hybrid default + phrase-aware sparse: measurement (2026-07-05)

Setup: `scripts/run_layout_eval.py --limit 100 -k 10` (TAT-QA, n=334 extractive queries, same
parameters as the Option-2 baseline), store rebuilt from scratch under **schema v2**.

| segment    |   n | DENSE | HYBRID (P3) | HYBRID (2026-06-24, pre-P3) |
|------------|----:|------:|------------:|----------------------------:|
| table      |  95 | 0.779 |       0.853 |                        0.853 |
| table-text | 110 | 0.855 |       0.945 |                        0.945 |
| text       | 129 | 0.938 |       0.953 |                        0.953 |
| OVERALL    | 334 | 0.865 |   **0.922** |                        0.922 |

Findings:

1. **Clean A/A.** Dense reproduces the June baseline digit-for-digit on a from-scratch schema-v2
   rebuild — B7, the read-assembly split, the method-bundle writer, and the hybrid-default flip
   moved retrieval by exactly zero. End-to-end regression-freedom, not just unit-level.
2. **P3 is a null result on TAT-QA** — hybrid-with-phrase-awareness equals pre-P3 hybrid exactly.
   TAT-QA *questions* are natural-language financial prose: no quoted spans, no `§` references,
   (almost) no dotted identifiers — the phrase path never fires. This also confirms the
   "plain queries compile byte-identically" claim in the strongest possible way.
3. **Where P3 pays** is identifier-style queries (`§6.4.2`, `"exact phrase"`, version numbers) —
   the technical-manual / ModusQ query style the behavior tests exercise. The roadmap's
   "TAT-QA (identifier-heavy)" measurement assumption was wrong for its question side; a real P3
   measurement needs a lexical-classified query slice over a manuals-style corpus (open item).

---

## P1 PR-1 — native table ingest + contextualized embedding: measurement (2026-07-05)

Setup as before (`--limit 100 -k 10`, n=334). New: tables ingest through the native `table_json`
extractor (chunks carry a real `Table`; cells persisted), and `EMBEDDING__CONTEXTUALIZE_TABLES`
embeds a table chunk as its header-contextualized rendering (`Table.contextual_text` — each value
bound to its row/column headers). Stored/BM25 text stays the grid.

| config                       | table | table-text | text  | overall |
|------------------------------|------:|-----------:|------:|--------:|
| plain-text tables, dense     | 0.779 |      0.855 | 0.938 |   0.865 |
| plain-text tables, hybrid    | 0.853 |      0.945 | 0.953 |   0.922 |
| **native ingest**, dense     | 0.768 |      0.864 | 0.938 |   0.865 |
| native ingest, hybrid        | 0.853 |      0.945 | 0.953 |   0.922 |
| native + **ctx**, dense      | 0.747 |      0.891 | 0.930 |   0.865 |
| native + ctx, hybrid         | 0.863 |      0.945 | 0.953 | **0.925** |

Findings:

1. **Native ingest is behavior-preserving**: hybrid digit-identical; dense a wash (−0.011 table /
   +0.009 table-text — the atomic table leaf vs the old char-split of oversize tables).
2. **Contextualized embedding is a small, mixed lever at the retrieval level**: on the shipped
   hybrid default +0.010 table / +0.003 overall; on pure dense it helps table-text (+0.027) but
   hurts pure-table (−0.021), netting flat. Hybrid had already closed most of the table
   *retrieval* gap. The flag ships **default-off** (an embed-time index variant, fingerprinted).
3. **The big measured table penalty is reader-side** (attribution 0.88 vs 0.99, F1 0.51 vs 0.60)
   — PR-2's territory (structured table view + cell citations for the reader). PR-1's real
   deliverable is the substrate PR-2 needs: real `Table`s with persisted cells flowing through
   ingest → chunks → hydrate.
4. Eval-script fix along the way: since P2 made hybrid the Settings default, the script's
   "DENSE" leg (which passed no spec) silently ran hybrid — it now pins an explicit dense-only
   spec.

---

## P1 PR-2 — structured table view for the reader: measurement (2026-07-06)

The reader (and the grounding judge, kept in lockstep) now sees a TABLE chunk as its
header-contextualized rendering (`passage_text` / `Table.contextual_text`) instead of the raw grid;
`table_view ∈ {structured, text}` is pinned explicitly on both components in the eval (never
inherited). Setup: native store, hybrid retrieval (pinned), `gpt-4o-mini`, n=334.

| segment    | F1 text→struct | EM          | attrib          | cite        |
|------------|----------------|-------------|-----------------|-------------|
| table      | 0.532→0.497    | 0.389→0.368 | **0.874→0.895** | 0.732→0.711 |
| table-text | 0.577→0.560    | 0.427→0.400 | **0.927→0.945** | 0.724→0.706 |
| text       | 0.577→0.589    | 0.333→0.341 | 0.984→0.984     | 0.868→0.876 |
| OVERALL    | 0.564→0.553    | 0.380→0.368 | **0.934→0.946** | 0.782→0.773 |

Findings:

1. **Attribution — the measured penalty PR-2 targets — improves on exactly the penalized
   segments** (table +0.021, table-text +0.018, overall +0.012), with the text segment as a clean
   control (0.984 both). The text-view leg reproduces the June baseline (0.874/0.532 vs 0.88/0.51)
   on the new native store — a valid A/A.
2. **F1/EM tick down ~0.01–0.035 on table segments** — at n=95/segment with gpt-4o-mini's
   ±0.02–0.03 nondeterminism both movements sit at the noise edge; no reliable answer-quality
   cost, no reliable gain. A larger-n run would resolve; not spent here.
3. Ships with ``structured`` as the default view (the principled representation; the judge always
   renders what the reader read). ``text`` remains one pinned config away for baselines.
4. Keyless-run footgun worth remembering: ``OPENAI_LLM_KEY`` in ``.env`` reaches Settings fields,
   **not** ``os.environ`` — an unexported key makes every LLM call fail silently through
   ``_safe_answer`` (F1 0.000 / attrib 1.000 across the board is the signature).

---

## P4 — reranker sweep → the quality profile (2026-07-06)

Setup: TAT-QA source-hit@10, n=334, native store, every leg's spec pinned (incl. `top_n`).

| leg                        | table | table-text | text  | overall |
|----------------------------|------:|-----------:|------:|--------:|
| DENSE                      | 0.768 |      0.864 | 0.938 |   0.865 |
| HYBRID (shipped default)   | 0.853 |      0.945 | 0.953 |   0.922 |
| HYBRID + cross_encoder (uncapped) | 0.895 | 0.964 | 0.946 | 0.937 |
| HYBRID + cross_encoder (top_n=20) | **0.926** | **0.964** | 0.946 | **0.946** |
| HYBRID + llm_judge (top_n=20)     | 0.874 | 0.936 | 0.953 |   0.925 |

Findings:

1. **The local cross-encoder wins decisively** (+0.024 overall, +0.073 table over hybrid) and the
   **LLM judge is noise-level** (+0.003) while costing an API call per query — the offline model
   beats it on every axis. Fits the offline differentiator.
2. **Capping helps quality, not just cost**: uncapped, the CE reranks the whole fused shortlist
   (~60 hydrated candidates) and promotes junk from the deep tail where it's miscalibrated
   (table 0.895); capped to the top-20 it works only where the first pass is already good
   (0.926) — and runs ~2.5× faster (26 min vs ~70 min for 334 queries on this CPU;
   `CrossEncoderReranker` now has `top_n=20`, parity with `llm_judge`).
3. **The table retrieval gap is essentially closed**: 0.926 vs 0.946 text (was 0.779 vs 0.938
   dense-only in June).
4. **The quality profile** = the shipped hybrid default + `reranker: {class_name: cross_encoder,
   top_n: 20}` (README snippet). The lean default stays reranker-free: ~2–4s/query of CPU rerank
   is a real cost; the profile is one config edit.

---

## PP-6 — table_lookup: deterministic numeric answers from cells (2026-07-06)

The production pattern's "route structured-data questions to tooling, not the reader" — in-library:
a `table_lookup` reasoner computes change / %-change / sum / average over one row's year-columns
straight from the persisted `table_cells` (accounting-format parsing, later-minus-earlier by year
value regardless of column order), cites the table chunk, and DELEGATES anything it can't resolve
confidently (cross-row, nested, text-sourced derivations). Eval: the TAT-QA **arithmetic** slice
(n=253 — previously filtered out entirely), numeric EM at gold precision (2dp), specs fully pinned.

| leg                                        | attempted | EM attempted | EM overall | LLM calls |
|--------------------------------------------|----------:|-------------:|-----------:|----------:|
| table_lookup only (no fallback)            | 95 (37.5%)|    **0.432** |      0.162 |     **0** |
| reader only (single_hop, structured tables)|       253 |        0.261 |      0.261 |       253 |
| **composed: table_lookup → reader fallback**|      253 |        0.304 |  **0.304** |       158 |

Findings:

1. **Where the deterministic path fires, it is 1.65× more accurate than the reader** (0.432 vs
   0.261) at zero cost — and it never guesses: the 37.5% attempt-rate is the conservative resolve
   gate abstaining on derivations outside its op set.
2. **The composition beats the reader on accuracy AND cost**: +4.3 EM pts overall while cutting
   LLM calls by 37.5%. This is the shipped default for the component
   (``fallback: single_hop``); ``fallback: null`` is the LLM-free / abstaining posture.
3. Levers left on the table (recorded, not spent): pct-change sign conventions (TAT-QA golds are
   inconsistent — some magnitude-based), count questions, cross-row derivations, row-matching
   recall. The reader's own 0.261 confirms LLM grid arithmetic is weak — routing was the right
   call.

---

## S1 — query-understanding sweep → the routed-profile verdict (2026-07-13)

Setup: TAT-QA source-hit@10, n=334, native store (cached index), every leg's spec pinned; the
structural classifier labels the set 33 lexical / 301 semantic (the same labels the router
dispatches on, so the slices measure exactly what each route would receive). LLM legs
(`multi_query` / `hyde` as the hybrid's dense arm) through gpt-4o-mini; `ROUTED` is composed from
the measured per-slice winners (ties to the cheaper leg): lexical → HYBRID, semantic → MQ+SPARSE.
Runner: `scripts/run_layout_eval.py --s1 --limit 100 -k 10`.

| leg                      | table | table-text | text  | lexical | semantic | overall |
|--------------------------|------:|-----------:|------:|--------:|---------:|--------:|
| HYBRID (shipped default) | 0.853 |      0.945 | 0.953 |   0.970 |    0.917 |   0.922 |
| HYBRID+CE (P4 profile)   | 0.926 |      0.964 | 0.946 |   0.970 |    0.944 | **0.946** |
| W-SPARSE-1.5             | 0.863 |      0.955 | 0.961 |   0.970 |    0.927 |   0.931 |
| W-SPARSE-2               | 0.863 |      0.945 | 0.961 |   0.970 |    0.924 |   0.928 |
| W-SPARSE-3               | 0.884 |      0.955 | 0.961 |   0.970 |    0.934 |   0.937 |
| MQ+SPARSE                | 0.874 |      0.955 | 0.953 |   0.939 |    0.930 |   0.931 |
| HYDE+SPARSE              | 0.874 |      0.936 | 0.946 |   0.970 |    0.917 |   0.922 |
| ROUTED                   | 0.895 |      0.955 | 0.953 |   0.970 |    0.934 |   0.937 |
| ROUTED+CE                | 0.926 |      0.964 | 0.946 |   0.970 |    0.944 | **0.946** |

Findings:

1. **Clean A/A**: HYBRID and HYBRID+CE reproduce the P2–P4 numbers digit-for-digit — the S1
   tooling (per-query hits, the sweep mode) moved nothing.
2. **The cross-encoder subsumes every retrieval-arm gain**: ROUTED+CE is digit-identical to
   HYBRID+CE on every column. Once the reranker sees the shortlist, how the shortlist was
   assembled (weighted, multi-query, routed) stops mattering at this corpus's margins. **The
   quality profile stays hybrid + CE** (P4, reconfirmed) — routing under it is pure cost.
3. **Below the CE tier the routing story is real but nuanced**: `multi_query` wins the semantic
   slice (+0.013) while *hurting* the lexical one (−0.031) — routing protects exactly that split
   (ROUTED: +0.015 overall, +0.042 table over hybrid, at ~1 LLM call per semantic query). But
   **W-SPARSE-3 matches ROUTED LLM-free** (0.937): on a corpus that rewards exact-token matching,
   up-weighting the sparse arm buys the same lift for nothing. The weights knob (S1 PR-2) is the
   cheap first lever; tune per corpus with `--s1`.
4. **HyDE is a null result on TAT-QA** (0.922 = hybrid; semantic slice identical). The
   fake-answer probe adds no recall where hybrid already has the vocabulary; its natural test is
   the recall-limited pool setting — P7's composition work, not spent here.
5. **The lexical slice still doesn't discriminate** (n=33, flat 0.970 on every leg that keeps a
   real sparse arm) — the P3 open item stands: a manuals-style corpus with identifier-heavy
   queries is the setting where the lexical route (and P3's phrase path) can actually be measured.
6. **Pool regression skipped, deliberately**: nothing shipped changes (default hybrid and the
   hybrid+CE profile are untouched), so there is no regression surface. MQ/HyDE over the pool
   belongs to P7 (sub-questions through hybrid + pooled-evidence rerank).

Verdict: S1's shipped deliverable is the *measured decision* plus the machinery — the routed
pipeline, both bridge retrievers, weighted RRF, and the `--s1` sweep are all in place and one
config edit away, with the README documenting when routing pays (no-reranker deployments on
mixed-form query sets) and when it doesn't (under the CE).

---

## P7a — sub-questions through hybrid + HyDE on the pool (2026-07-13)

Setup: the Phase-2.5/3 protocol — HotpotQA over the shared ~10K-passage pool (`--corpus pool
--corpus-limit 1000`), decomposition reasoner, gpt-4o-mini reader (also the hypothesis model),
n=200, every leg's retrieval spec pinned via `--retrieval-spec`. The June pool cache predated B7
and was refused at open (`schema_version 1 != 2` — the guard doing its job); the index was
rebuilt from scratch under schema v2, with the DENSE leg as the A/A.

| leg                        |   hit |    F1 |    EM |
|----------------------------|------:|------:|------:|
| DENSE (pinned; June's arm) | 0.475 | 0.607 | 0.485 |
| HYBRID (P7a)               | 0.486 | 0.620 | 0.495 |
| HYDE+SPARSE                | **0.514** | **0.665** | **0.525** |

Findings:

1. **A/A holds on the v2 rebuild**: dense lands inside the June band (hit 0.481–0.508, F1
   0.610–0.633 across the recorded runs) — the schema-v2 rebuild moved nothing beyond the
   documented ±0.02–0.03 nondeterminism.
2. **Hybrid sub-questions are a within-noise nudge on wiki-QA** (+0.011 hit / +0.013 F1 /
   +0.010 EM) — consistent with Option 1 (+0.015 F1); hybrid's measured home stays layout/table
   corpora. It is the shipped default and free at query time; nothing here argues otherwise.
3. **HyDE is the first retrieval lever to clear the noise band on the pool**: +0.039 hit /
   +0.058 F1 / +0.040 EM over dense. The mechanism matches the wall's shape (Phase 2.5: the
   2nd-hop passage resembles the *answer*, not the question) — the fake answer embeds where the
   missing hop lives. Together with S1 (HyDE null on TAT-QA), the split is exactly as predicted:
   HyDE pays where recall binds, not where precision does. Cost: one hypothesis call per
   sub-question.
4. The MOTHRAG F1 gap narrows from −0.148 (June's best, 0.633) to −0.116 — with the mini reader.
