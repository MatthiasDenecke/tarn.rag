# Component catalog

Extractors, chunkers, retrievers, reasoners — every pluggable part of tarn.rag is a **Component**
(the [`bausatz`](https://pypi.org/project/bausatz/) framework), instantiated from a spec under
`Settings.components`. A spec is a mapping whose `class_name` selects the component by its
registered tag; the remaining keys are that component's own config, including nested child specs:

```yaml
retrieval_pipeline:
  class_name: retrieval_pipeline          # ← the tag
  retrievers: [{class_name: dense}]       # ← child component specs
  fuser: {class_name: identity}
```

A spec is validated as a **whole tree** when its engine builds it: every nested child is resolved to
its concrete config and checked in the same pass, so an unknown option or a misplaced child (say, a
fuser in the `retrievers` list) fails at startup with its full path (`retrievers.0`), never on the
first query. Each slot only accepts components of its own family — the sections below.

Tags marked **(default)** are what you get without configuring anything.

## Ingestion

### Pipeline & stages (`components.ingestion_pipeline`)

| Tag | Component | Role |
|---|---|---|
| `pipeline` **(default)** | `Pipeline` | The stage DAG; default stages below, in order. |
| `LoadAndParse` | `LoadAndParseStage` | Routes each `source_kind` to an extractor; produces a `StructuredDocument`. |
| `Enrich` | `EnrichStage` | Runs document-phase enrichers (default: none). |
| `CleanAndNormalize` | `CleanAndNormalizeStage` | Normalizes line endings, strips control characters, collapses whitespace (`collapse_whitespace`) in the structured document — every span remapped, code blocks and tables kept verbatim. The cleaned text is what gets stored, chunked, and embedded. |
| `Chunk` | `ChunkStage` | Splits via its configured chunker (default `structure_aware`). |
| `Embed` | `EmbedStage` | Embeds chunks with the shared embedder. |

### Extractors (children of `LoadAndParse`, keyed by `source_kind` route)

| Tag | Component | Notes |
|---|---|---|
| `plain_text` | `PlainTextExtractor` | |
| `markdown` | `MarkdownExtractor` | |
| `html` | `HtmlExtractor` | BeautifulSoup — `parsers` extra. |
| `pdf_text` | `PdfTextExtractor` | pdfplumber, the fast tier — `parsers` extra. |
| `docling` | `DoclingExtractor` | High-fidelity layout-aware PDF — `docling` extra (heavy). |
| `table_json` | `TableJsonExtractor` | Tables from a JSON representation. |

A per-document `metadata['extractor']` override picks one inline.

### Chunkers (child of `Chunk`)

| Tag | Component | Notes |
|---|---|---|
| `structure_aware` **(default)** | `StructureAwareChunker` | Follows the `StructuredDocument` (headers, tables, geometry); emits `ChunkProvenance`. Leaves are the retrieval units; section parents are stored for `auto_merge`, never embedded or searched. |
| `recursive` | `RecursiveCharacterChunker` | Classic size/overlap character splitting. |

### Enrichers (children of `Enrich`)

| Tag | Component | Notes |
|---|---|---|
| `acronyms` | `AcronymEnricher` | Annotates acronym expansions on the document. |

## Retrieval (`components.retrieval_pipeline`)

### Searchers (the spec root)

| Tag | Component | Notes |
|---|---|---|
| `retrieval_pipeline` **(default)** | `RetrievalPipeline` | retrievers → fuser → hydrate → optional merger → optional reranker → `top_k`. |
| `routing_retrieval_pipeline` | `RoutingRetrievalPipeline` | A `QueryClassifier` dispatches per `query_type` to per-type pipelines. |

### Retrievers

| Tag | Component | Notes |
|---|---|---|
| `dense` **(default)** | `DenseRetriever` | Vector KNN (sqlite-vec / pgvector). |
| `sparse` **(default)** | `SparseRetriever` | BM25 (FTS5 / Postgres). |
| `hyde` | `HydeRetriever` | Embeds a hypothetical generated answer (needs the LLM). |
| `multi_query` | `MultiQueryRetriever` | Fans out LLM query reformulations (needs the LLM). |

License/scope filtering happens *inside* the retrievers as a SQL pre-filter, over-fetching to
backfill past dropped chunks.

### Fusers

| Tag | Component | Notes |
|---|---|---|
| `rrf` **(default)** | `RRFFuser` | Reciprocal-rank fusion; ties break `(score desc, chunk_id asc)`. |
| `identity` | `IdentityFuser` | Pass-through — for single-retriever pipelines. |

### Mergers (optional slot)

| Tag | Component | Notes |
|---|---|---|
| `auto_merge` | `AutoMerger` | Recombines ≥ `min_siblings` sibling chunks into their section parent (`parent_chunk_id` provenance), repeating up the section tree; a parent must pass the query's license/scope/grounding filter. |

### Rerankers (optional slot)

| Tag | Component | Notes |
|---|---|---|
| `cross_encoder` | `CrossEncoderReranker` | Local ONNX cross-encoder (`settings.rerank`); lazy-loaded. |
| `llm_judge` | `LlmJudgeReranker` | LLM-scored relevance. |

### Query classifiers (for `routing_retrieval_pipeline`)

| Tag | Component |
|---|---|
| `generic` | `GenericQueryClassifier` |
| `intent` | `IntentQueryClassifier` |
| `structural` | `StructuralQueryClassifier` |

### License policies (`components.license_policy`)

| Tag | Component | Notes |
|---|---|---|
| `default_license` **(default)** | `DefaultLicensePolicy` | Purpose → permitted license classes; `third_party_copyrighted` never permitted (ModusQ §5.6). |

## Generation (`components.generation_pipeline`)

### Pipeline

| Tag | Component | Notes |
|---|---|---|
| `generation_pipeline` **(default)** | `GenerationPipeline` | reasoner → grounding check → evidence assembly. |

### Reasoners

| Tag | Component | Notes |
|---|---|---|
| `decomposition` **(default)** | `DecompositionReasoner` | Decompose → per-sub-question retrieval → synthesis; best on average across multi-hop benchmarks, costs more LLM calls. |
| `single_hop` | `SingleHopReasoner` | One retrieval, one LLM call — the cheap path. |
| `iterative` | `IterativeReasoner` | Retrieve ↔ read loop until answerable. |
| `grounded_retrieval` | `GroundedRetrievalReasoner` | Retrieval interleaved with grounding feedback. |
| `answerability` | `AnswerabilityGateReasoner` | Gates a wrapped reasoner on whether evidence can answer at all. |
| `table_lookup` | `TableLookupReasoner` | Deterministic numeric answers straight from persisted table cells — no LLM arithmetic. |

### Evidence assemblers

| Tag | Component | Notes |
|---|---|---|
| `provenance` | `ProvenanceAssembler` | Builds the proof tree + evidence from chunk provenance. |

### Grounding checkers

| Tag | Component | Notes |
|---|---|---|
| `heuristic_grounding` | `HeuristicGroundingChecker` | Fast, LLM-free checks. |
| `llm_grounding` | `LLMGroundingChecker` | LLM-verified support. |
| `cascading_grounding` | `CascadingGroundingChecker` | Runs a checker cascade (e.g. heuristic first, LLM to confirm). |

A worked generation spec — cascading grounding with abstention — ships in
[`examples/console.config.json`](../../examples/console.config.json).

## Conversation (`components.conversation_memory`)

Applied by the `TarnRag` facade, not by an engine. Off unless `enabled: true`.

### Memory

| Tag | Component | Notes |
|---|---|---|
| `conversation_memory` **(default, disabled)** | `ConversationMemory` | Reads the last `max_turns` turns of the conversation, has its contextualizer make the question standalone, and records `ask` / `retrieve` turns. Fields: [settings reference](settings.md#conversation_memory). |

### Contextualizers (child of `conversation_memory`)

| Tag | Component | Notes |
|---|---|---|
| `condense` **(default)** | `CondenseContextualizer` | One LLM call that rewrites a follow-up into a standalone question (`max_tokens`, default 256, caps the call). Not called on a conversation's first turn. Falls back to the question as typed — reported as a warning — if there is no LLM or the reply is unusable. |
