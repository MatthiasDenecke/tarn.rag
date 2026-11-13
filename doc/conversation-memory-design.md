# Conversation memory — design

**Status:** ✅ **Implemented** (2026-11-13) — approved as written; slices 1–7 (§12) are all built. Code:
`tarnrag/conversation/` (`memory.py` = `ConversationMemory`; `components/contextualizer.py` =
`QueryContextualizer` + `condense`; `context.py`, `types.py`), `ConversationTurn` + `ConversationStore`
in `tarnrag/contracts/`, the `conversation_turns` table in `storage/repository/base.py`, the wiring in
`tarnrag/tarnrag.py`, and the console commands. Two corrections to the approved text, both of fact:
Part II has **eight** config files, not nine (rung 01 has none of its own), and the facade tests live
in their own file, `tests/test_tarnrag_conversation.py`.

The design for a **short-term conversational context layer**: tarn.rag today answers every question in
isolation, so a follow-up such as *"which oil grade does it require?"* is searched and read exactly as
typed. This doc fixes the seams before any code, the way `doc/generation-architecture-design.md` did for
generation. Section 9 records every decision and who made it.

## 0. Requirements & constraints (locked)

1. **Resolve follow-ups against the recent conversation.** A question that depends on earlier turns
   (a pronoun, an omitted subject, *"and for the Ariel one?"*) is turned into a **standalone question**
   before anything else runs. Retrieval, every reasoner, the grounding check and the proof tree then
   work on that question, unchanged.
2. **Short-term.** The layer remembers the last *N* turns of one conversation. It is not long-term
   memory: no cross-conversation facts, no summaries, no user profile (§10).
3. **Opt-in, and off by default.** Switched by one flag in the config
   (`components.conversation_memory.enabled`). With the flag off the facade takes today's code path
   exactly: no store read, no LLM call, no write.
4. **The engines and the wire contracts do not change.** `IngestionEngine`, `RetrievalEngine`,
   `GenerationEngine`, the `Query` contract and `GenerationResult` are untouched. The retrieval core
   stays LLM-free by default and C++-portable; the one-way rule `generation → retrieval` is not
   weakened, and neither engine learns that conversations exist.
5. **Grounding is not weakened.** Conversation history never reaches the reader. An answer is still
   built only from retrieved passages and still verified against them; history influences *which
   question is asked*, never *what counts as evidence*.
6. **Nothing is hidden.** A rewrite changes what is searched and answered, so the question as typed and
   the question as executed are both recorded and retrievable (`TarnRag.history()`), and a rewrite that
   could not be made is reported, not silently skipped.
7. **REST stays out.** The layer is library API on the `TarnRag` facade; tiqtasq.backend wraps it.

## 1. Where the layer sits

```
  your code · the console · tiqtasq.backend
        │
        ▼
  TarnRag  (facade + composition root)
        │  1. contextualize   ┌──────────────────────────────┐
        ├────────────────────►│ tarnrag/conversation/   NEW  │──► LanguageModel (core Resource)
        │  4. remember        │  ConversationMemory          │──► ConversationStore (port)
        │                     └──────────────────────────────┘         ▲ implemented by
        │  2. standalone Query                                          │
        ├──► RetrievalEngine  (retrieve / explain)        DocumentRepository — the one store
        └──► GenerationEngine (ask)  ── 3. answer         (+ one new table: conversation_turns)
```

`tarnrag/conversation/` is a new top-level package beside `ingestion/`, `retrieval/` and `generation/`.

**Dependency rule.** `conversation → core + contracts`, nothing else. It imports neither `retrieval`
nor `generation`, and neither of them imports it: the contextualizer works on plain question text, and
the facade — the only place that knows all four packages — builds the `Query`. The repository
implements the layer's storage port, as it implements every other port (`storage → contracts`).

**Consequence (accepted, §9 D5).** Memory is a facade feature. A caller using a bare
`GenerationEngine.create()` or `RetrievalEngine.create()` gets no memory unless they compose the layer
themselves; the eval harnesses and benchmark runners, which drive engines and pipelines directly, stay
stateless and are unaffected.

## 2. The flow

```
retrieve(text)   contextualize → RetrievalEngine.search(Query)   → remember (kind=retrieve)
explain(text)    contextualize → RetrievalEngine.explain(Query)     (dry run: nothing remembered)
ask(text)        contextualize → GenerationEngine.answer(Query)  → remember (kind=ask, + answer)
```

**Contextualize** (read-only):

1. Read the window: the last `max_turns` turns of the current conversation, oldest first.
2. Empty window → the question is used as typed. **No LLM call** — the first turn of a conversation
   costs nothing extra.
3. Otherwise one LLM call rewrites the question into a standalone one (§3.2). The facade builds the
   `Query` from the standalone text and attaches one `Annotation`
   (`producer="condense"`, `type="contextualized"`, `value={"original": …, "turns_used": n}`,
   `deterministic=False`) — the channel the design already uses for LLM findings on a query, so a
   rewrite is flagged as generated, never silently trusted. Reasoners derive sub-queries with
   `replace`, so the annotation rides along to every search.

**Remember** runs only after the operation returned; an operation that raises records nothing.

| Operation | Reads memory | Writes a turn | What the turn holds |
|---|---|---|---|
| `ask` | yes | yes | question as typed, standalone question, answer, `grounded`, `abstained` |
| `retrieve` | yes | yes | question as typed, standalone question (no answer) |
| `explain` | yes | **no** | — a diagnostic shows what `retrieve` would search without changing state |

Every `ask` turn stores its answer, including ungrounded and refused ones, with their flags (§9 D8).

## 3. The seams

All new pluggable parts are **Components** (bausatz), pure strategies with resources injected at call
time through a context — the same shape as `RetrievalContext` / `GenerationContext`.

### 3.1 `ConversationMemory` — the container Component

Built from `Settings.components[CONVERSATION_MEMORY]`. Owns the window policy and delegates the
rewrite to its child.

```python
class ConversationMemory(Component):                       # tarnrag/conversation/memory.py
    class Config(Component.Config):
        class_name: Literal["conversation_memory"] = "conversation_memory"
        enabled: bool = False              # the on/off switch (off = today's stateless behaviour)
        session_id: str = ""               # "" → a fresh conversation per TarnRag.open(); set → resume it
        max_turns: int = Field(6, ge=1)    # the short-term window: how many recent turns are read
        max_answer_chars: int = Field(600, ge=1)   # each remembered answer is cut to this in the window
        contextualizer: QueryContextualizer.Config = Field(default_factory=CondenseContextualizer.Config)

    async def contextualize(self, question: str, session_id: str, ctx: ConversationContext) -> Contextualization: ...
    async def remember(self, turn: ConversationTurn, ctx: ConversationContext) -> ConversationTurn: ...
    async def history(self, session_id: str, ctx: ConversationContext, last: int | None = None) -> list[ConversationTurn]: ...
    async def forget(self, session_id: str, ctx: ConversationContext) -> int: ...

@dataclass
class ConversationContext:                                 # tarnrag/conversation/context.py
    store: ConversationStore        # the port (§4)
    llm: LanguageModel | None       # the rewriter; None → no rewrite possible (reported, §6)
```

The component is stateless: it holds no turns and no session. The session id is passed in on every
call, so the component and the store are already keyed per conversation — a later multi-conversation
API needs no redesign here (§10).

### 3.2 `QueryContextualizer` — how history is used

```python
class QueryContextualizer(Component):                      # tarnrag/conversation/components/contextualizer.py
    @abstractmethod
    async def contextualize(
        self, question: str, history: list[ConversationTurn], ctx: ConversationContext
    ) -> Contextualization: ...

class CondenseContextualizer(QueryContextualizer):         # class_name: "condense"
    class Config(QueryContextualizer.Config):
        class_name: Literal["condense"] = "condense"
        max_tokens: int = Field(256, ge=1)   # cap on the rewrite call — a rewrite is one sentence
```

`condense` makes one `LanguageModel.complete` call. The prompt lays out the window as numbered turns
(each turn's **standalone** question, so references resolve transitively across turns; its answer, cut
to `max_answer_chars` and marked when it was unverified or a refusal; a `retrieve` turn is shown as a
search with no answer) followed by the latest question. The instruction: resolve references only, do
not answer, add no facts that are not in the conversation, keep identifiers and names verbatim, return
the question unchanged if it is already self-contained. The reply is a single JSON object
`{"standalone_question": "…"}`, parsed with the shared `core.parsing.extract_json`.

```python
@dataclass(frozen=True)
class Contextualization:                                   # tarnrag/conversation/types.py
    original: str                  # the question as typed
    standalone: str                # the question as executed (== original when nothing was rewritten)
    turns_used: int = 0            # how many remembered turns the contextualizer saw
    producer: str = ""             # the contextualizer's class_name ("" when none ran)
    fallback: str | None = None    # why the question was used as typed although history existed
```

Other strategies (a deterministic rule-based resolver, history-aware query expansion) are later
components on this seam; none is built now.

## 4. Persistence

Turns are persisted in the **one store**, the `DocumentRepository` (§9 D2), behind a narrow port so the
conversation package depends on an interface, not on the repository (ISP, like `RetrievalStore`).

```python
@dataclass(frozen=True)
class ConversationTurn:                                    # tarnrag/contracts/dtos.py
    session_id: str
    kind: str                           # "ask" | "retrieve"
    question: str                       # as typed
    standalone_question: str            # as executed
    answer: str | None = None           # None for a retrieve turn
    grounded: bool | None = None        # None for a retrieve turn
    abstained: bool | None = None
    ordinal: int = -1                   # position in the conversation, assigned by the store
    created_at: datetime | None = None  # assigned by the store

class ConversationStore(ABC):                              # tarnrag/contracts/ports.py
    async def append_turn(self, turn: ConversationTurn) -> ConversationTurn: ...
    async def conversation_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]: ...
        # oldest first; `limit` keeps the most recent N (the window); None = the whole conversation
    async def delete_conversation(self, session_id: str) -> int: ...   # rows removed
```

**The table** — `conversation_turns`, defined once in `storage/repository/base.py` with shared Core
CRUD (no dialect hook is needed, so SQLite and Postgres behave identically):

| Column | Type | Notes |
|---|---|---|
| `session_id` | TEXT | part of the PK |
| `ordinal` | INTEGER | part of the PK; 0-based position within the conversation |
| `kind` | TEXT | `ask` \| `retrieve` (CHECK) |
| `question` | TEXT | as typed |
| `standalone_question` | TEXT | as executed |
| `answer` | TEXT NULL | |
| `grounded`, `abstained` | INTEGER NULL | 0/1 |
| `created_at` | TIMESTAMP | server default, like `job_status` |

- **Not part of the §8 index.** Like `job_status`, it is a Python-side table living in the same store.
  The C++ reader reads only the tables it knows.
- **`SCHEMA_VERSION` stays `"2"`.** That version is bumped only for an *incompatible* §8 change; this
  is an additive table that `connect()`'s `create_all` adds to an existing store. No store has to be
  rebuilt, and no existing table or column changes.
- **Retention: keep all, read the window** (§9 D7). Turns outside the window are never read by the
  rewriter but stay in the table until `forget()` deletes that conversation. Consequence, accepted:
  question and answer text accumulates inside the index file, so a copied or shipped index carries its
  conversations with it.
- **A conversation is sequential.** `append_turn` assigns the next ordinal in a single statement; two
  writers appending to the *same* conversation at the same instant are not supported (the PK rejects
  the loser). Different conversations never interfere.
- **Deleting a document does not edit conversations.** A remembered answer is text, not a reference;
  it can outlive the document it came from. `forget()` is the remedy.

## 5. Sessions

One `TarnRag` instance is one conversation (§9 D6). Its identity on `open()` (§9 D10):

1. the `session_id` passed to `TarnRag(settings, session_id=…)`, else
2. `components.conversation_memory.session_id` from the config, else
3. a freshly generated id — a new, empty conversation.

So a new process starts clean by default, and resumes a stored conversation only when asked to. With
memory disabled there is no session (`tarn.session_id is None`).

Accepted consequences: one facade instance serves one conversation at a time; a script that asks
unrelated questions through one memory-enabled `TarnRag` will have them read as one conversation
(call `new_conversation()` between them, or leave memory off).

## 6. Facade and console surface

**`TarnRag`** — existing signatures are unchanged; the constructor gains one optional argument and
three methods are added. All return an `Outcome`, as everything on the facade does.

```python
TarnRag(settings, llm=None, session_id=None)

tarn.session_id                      # str | None — the current conversation (None when memory is off)
await tarn.history(last=None)        # Outcome[list[ConversationTurn]] — this conversation, oldest first
await tarn.new_conversation()        # Outcome[str | None] — start a fresh one; the old transcript stays stored
await tarn.forget()                  # Outcome[int] — delete this conversation's turns; value = how many
```

`retrieve` / `explain` / `ask` behave as in §2. How a caller learns what a question became (§9 D12):
`history()` returns each turn's `question` and `standalone_question`; `explain` shows it directly,
because the returned `SearchTrace.query` is the executed query and carries the `contextualized`
annotation. `ConversationTurn` is exported from `tarnrag`.

**The LLM.** The rewrite uses the same `LanguageModel` as generation (`Settings.llm`, or the injected
one) — one model, built lazily on first need and closed by the facade if the facade built it. A
retrieval-only session with memory on therefore needs no key until its first follow-up.

**When a rewrite cannot be made** (follows existing precedent, §9 P1):

| Situation | Behaviour |
|---|---|
| Memory on, history exists, but no LLM is available (no key, none injected) | The question is used as typed; the `Outcome.report` carries a WARNING naming the missing key. (`ask` then fails for lack of a key, as it does today.) |
| The model's reply is not the expected JSON, or the rewritten question is empty | The question is used as typed; WARNING in the report. |
| The LLM call itself raises (transport, auth) | The error propagates — a misconfiguration fails loudly, as `multi_query` / `hyde` do. |
| `history` / `forget` / `new_conversation` while memory is off | An empty value plus a WARNING that memory is disabled — not a silent no-op. |

**Console** — three new commands, and the rewrite made visible:

| Command | Effect |
|---|---|
| `history` | The current conversation: each turn as typed, as understood, and its answer. |
| `new` | Start a fresh conversation (the previous transcript stays in the store). |
| `forget` | Delete the current conversation's turns from the store. |

The start-up banner states whether memory is on and, if so, the session id. After an `ask` or
`retrieve` whose question was rewritten, the console prints one line, `↳ understood as: …`. `explain`
shows the executed query plus an `asked: …` line with the original wording.

## 7. Configuration

A fifth recognized key under `Settings.components`, default-filled by `_fill_default_components` like
the other four, so a `Settings` is always self-complete — and the default is **disabled**:

```yaml
components:
  conversation_memory:
    class_name: conversation_memory
    enabled: true            # ← the on/off switch
    session_id: ""           # "" = a fresh conversation per run; set an id to resume one
    max_turns: 6             # the short-term window
    max_answer_chars: 600    # each remembered answer is cut to this before the rewriter sees it
    contextualizer:
      class_name: condense
      max_tokens: 256
```

The spec validates as a whole tree with the rest (an unknown option, `max_turns: 0`, or a foreign
child in the `contextualizer` slot fails at start-up with its path). Turning memory on or off is a
one-line diff between two configs.

**Existing Part II configs.** The repo rule is that Part II configs rely on no library default. The
eight `examples/part_ii/example_*/config.yaml` files therefore each gain one explicit line pinning
memory **off** (`conversation_memory: { class_name: conversation_memory, enabled: false }`). Every rung
gets the same line, so the diffs between rungs — the lessons — are unchanged, as are all their results.

## 8. How it composes with what exists

- **Reasoners** (`single_hop`, `decomposition`, `iterative`, `grounded_retrieval`, `table_lookup`,
  `answerability`): unchanged. They receive a standalone `Query`.
- **Answerability gate:** benefits — it checks the exact-match cues of the *standalone* question, so an
  identifier restored by the rewrite (`TX-200`) is verified against the evidence.
- **Query routing:** the classifier runs on the standalone text, which is what its form heuristics
  assume. The extra annotation is inert to it.
- **License / purpose / grounding flags:** untouched; the facade builds the `Query` exactly as before,
  with a different `text`. A conversation is assumed to belong to one principal with one purpose —
  remembered text is not re-filtered when it is shown to the rewriter (§10).
- **Embedded vs. distributed:** the table and its CRUD are shared Core, so both dialects are covered by
  one implementation. As with sparse search, the suite exercises SQLite; the Postgres path is written
  and reviewed but needs a live Postgres to run.
- **Cost:** one extra LLM call per follow-up turn (none on a conversation's first turn, none with
  memory off), plus one indexed read and one insert.

## 9. Decisions

**Made by the project owner (2026-11-13)** — each with the drawback that was accepted:

| # | Decision | Chosen | Accepted drawback |
|---|---|---|---|
| D1 | How history is used | Rewrite the question into a standalone one | One extra LLM call per follow-up; a bad rewrite misleads the whole turn; the reader never sees earlier wording |
| D2 | Who owns the turns | Persisted in the `DocumentRepository` | A new table in the portable store; conversation text lives in the index file |
| D3 | What "short-term" means | The last *N* turns, answers cut to a character cap | A long conversation forgets early context abruptly |
| D4 | Which operations are conversation-aware | `ask`, `retrieve` and `explain` | A follow-up `retrieve` / `explain` now needs an LLM (only when memory is on and history exists) |
| D5 | Where the layer lives | Its own package, applied by the `TarnRag` facade | Standalone engine users get no memory |
| D6 | Which conversation a call belongs to | One implicit conversation per `TarnRag` | One facade instance cannot serve several conversations at once |
| D7 | Turns outside the window | Kept; only the window is read; deleted on request | Unbounded growth; text accumulates in the index file |
| D8 | What an `ask` turn stores | Question and every answer, with `grounded` / `abstained` flags | An unsupported or refused answer can steer the next rewrite |
| D9 | Config shape | A component spec with an explicit `enabled` flag | An `enabled` flag on a component is new in this codebase |
| D10 | Conversation on reopen | Fresh, unless a session id is configured or passed | One more knob |
| D11 | Which operations record a turn | `ask` and `retrieve`; `explain` is a dry run | `explain` and `retrieve` differ in their effect on state |
| D12 | How a rewrite is surfaced | Through `history()` (and natively in `explain`'s trace) | A UI makes a second call to show "understood as" |
| D13 | The examples' LLM | The shipped scripted model (`StaticLanguageModel`), offline | The rewrite and answers are scripted; the examples prove the plumbing and the retrieval difference, not live LLM behaviour |
| D14 | Where the examples live | A new Part III series | New series scaffolding |

**Settled by existing precedent** (not asked; say so if any should change):

| # | Choice | Precedent |
|---|---|---|
| P1 | Rewrite failures: parse failure or missing LLM → use the question as typed and report a WARNING; a raising LLM call propagates | `multi_query` / `hyde` degrade on an unusable reply and fail loudly on transport errors; the facade's `Report` exists for "carried on past it" |
| P2 | `SCHEMA_VERSION` is not bumped | It is bumped only for incompatible §8 changes; this is an additive non-§8 table |
| P3 | No LLM call on an empty window | The lean-by-default rule: pay only when the feature does work |
| P4 | The window shows earlier turns' *standalone* questions, and marks unverified / refused answers | Follows D8 (flags are stored so they can be shown); standalone questions make chained references resolvable |
| P5 | One shared `LanguageModel` for rewrite and generation | The facade is the composition root: resources are built once and injected |
| P6 | Part II configs pin `enabled: false` explicitly | "Part II configs are fully explicit — no reliance on library defaults" |
| P7 | Deleting a document leaves conversations alone | Turns hold text, not chunk references; `forget()` is the explicit remedy |

## 10. Out of scope (deliberately)

- **Long-term memory:** cross-conversation facts, rolling summaries, user profiles.
- **History in the reader prompt.** Excluded by requirement 5, not merely deferred.
- **Several conversations through one facade** (an explicit per-call session id). The component and the
  store are keyed by session id already, so this is an additive facade change if tiqtasq.backend needs it.
- **Retention policies:** pruning, idle expiry, per-tenant purge.
- **Memory for standalone engines.**
- **Per-turn permission re-checks** when remembered text is shown to the rewriter.
- **Measuring rewrite quality.** A conversational eval set (follow-up → gold standalone question, scored
  through the existing retrieval harness) is the natural next step, and the only honest way to tune
  `max_turns` or compare contextualizers.
- **Streaming**, and any change to `GenerationResult`.

## 11. Deliverables

**Code**

| Area | Change |
|---|---|
| `tarnrag/contracts/` | `ConversationTurn` (dtos), `ConversationStore` (ports), exports |
| `tarnrag/storage/repository/base.py` | `conversation_turns` table + the three port methods |
| `tarnrag/conversation/` (new) | `memory.py`, `context.py`, `types.py`, `components/contextualizer.py` |
| `tarnrag/core/engine/config.py` | `CONVERSATION_MEMORY` key, default-filled disabled |
| `tarnrag/tarnrag.py` | wiring (§2), `session_id`, `history` / `new_conversation` / `forget`, shared lazy LLM |
| `tarnrag/console.py` | `history` / `new` / `forget`, the banner line, `↳ understood as`, `asked:` in `explain` |
| `tarnrag/__init__.py` | export `ConversationTurn` |

**Tests** (all on SQLite + the in-memory queue; the facade tests use the model-free `hash` embedder and
an injected `StaticLanguageModel`, so they run in CI with no model and no key)

- `tests/storage/repository/` — append assigns ordinals; `limit` returns the most recent turns in
  chronological order; conversations are isolated; delete; turns survive a reconnect; an existing store
  without the table gains it on connect.
- `tests/conversation/` — the prompt carries the window; JSON parsing; unchanged-when-standalone; the
  two fallbacks; answer truncation; retrieve-turn rendering; empty window makes no LLM call; the window
  is the last *N*, oldest first; spec validation (bad values, foreign child) and round-trip.
- `tests/core/` — the fifth key is default-filled, disabled, and validates with the other defaults.
- `tests/test_tarnrag_conversation.py` — memory off is byte-for-byte today's behaviour (no turns written, no extra
  LLM call); memory on rewrites the follow-up that reaches retrieval and the reader; `ask` and
  `retrieve` record, `explain` does not; a failed operation records nothing; fresh by default, resumed
  by config or argument across two facade instances; `history` / `new_conversation` / `forget`; the
  WARNING paths; the switch read from a YAML file.
- `tests/test_console*.py` — the three commands, the banner, the `understood as` line.
- `tests/examples/test_part_iii.py` — the two example interactions (below).

**Examples** — `examples/part_iii/`, two entries over the store Part II's rung 08 already defines
(corpus-2 plus the four frame specs), with configs that differ in exactly one line:

- `example_01` — memory **off**. The follow-up is searched as typed.
- `example_02` — memory **on**. The same conversation, the follow-up rewritten.

The conversation (verified against real retrieval on 2026-11-13, top-4, model-free):

| Turn | As typed | Rank of the correct spec (`lubrication-spec`) |
|---|---|---|
| 1 | *Which frame is the TX-200 built on?* | n/a — `compressor-models` is rank 1 |
| 2, memory off | *Which oil grade does it require?* | **4** (a different frame's spec is rank 1) |
| 2, memory on | → *Which oil grade does the Cooper-Bessemer GMV frame require?* | **1** |

Both examples inject a scripted model (D13). To keep the demonstration honest, the script is bound to
what the pipeline hands it: the scripted rewriter takes its referent from the transcript it is shown
(so it only resolves "it" if memory really delivered the earlier turn), and the scripted reader answers
only from the passages it is shown. The verdict each example prints is the model-free retrieval check
above. The published numbers were re-measured from the finished examples (16 documents · 24 chunks;
reader depth 3: the follow-up as typed leaves `lubrication-spec` at rank 4, the rewrite puts it at 1).

**Documentation** (Diátaxis; every new page linked from its section index)

- `docs/tutorials/part-iii/` — `index.md` plus one page per example; linked from `docs/tutorials/index.md`,
  `docs/index.md` and `examples/README.md`; a card README in each example directory.
- `docs/how-to/enable-conversation-memory.md` — the recipe.
- `docs/reference/` — `settings.md` (the fifth key), `components.md` (the two tags),
  `console-commands.md` (the three commands).
- `docs/explanation/architecture.md` — the layer, its dependency rule, and why history never reaches
  the reader.
- `CLAUDE.md` — package layout, the facade surface, the config key.
- This document's status line is updated when the slices land.

## 12. Build slices

1. **Contracts + storage** — `ConversationTurn`, `ConversationStore`, the table and its CRUD, tests.
2. **The conversation package** — `QueryContextualizer` + `condense`, `ConversationMemory`, context and
   result types, tests.
3. **Settings** — the fifth key, default-filled disabled, tests.
4. **Facade** — the wiring, the session rules, the three new methods, tests.
5. **Console** — the commands and the visible rewrite, tests.
6. **Examples** — Part III's two entries and their smoke tests.
7. **Docs** — tutorials, how-to, reference, explanation, `CLAUDE.md`; the Part II config pins.

Each slice leaves the full suite green; slices 1–3 change no behaviour for any existing caller.
