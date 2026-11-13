# 2 · Conversation memory

> **The follow-up is made to stand on its own before anything else runs.**
> **Config:** `examples/part_iii/example_02/config.yaml` · **No API key needed** · reads the Part III store

One line changes from [example 01](01-a-conversation-without-memory.md):

```yaml
  conversation_memory:
    class_name: conversation_memory
    enabled: true           # ← was false
    session_id: ""          # "" = a fresh conversation per run
    max_turns: 6            # how many recent turns the rewriter is shown
    max_answer_chars: 600   # each remembered answer is cut to this first
    contextualizer:
      class_name: condense  # one LLM call: resolve references, nothing else
      max_tokens: 256
```

The other settings were already in example 01's config, inert. Nothing about retrieval or generation
is different.

## The first turn costs nothing extra

```text
you › Which frame is the TX-200 built on?
searched  Which frame is the TX-200 built on?  (as typed)
top 3     compressor-models  ·  lubrication-spec  ·  lubrication-ariel
HIT   the passage holding 'GMV' (compressor-models) is rank 1

answer: Cooper-Bessemer GMV frame
```

Identical to example 01. There is no earlier turn to resolve anything against, so no rewrite is
attempted and no extra model call is made. When the answer comes back, the turn is recorded.

## The follow-up is rewritten, then searched

```text
you › Which oil grade does it require?
searched  Which oil grade does the Cooper-Bessemer GMV frame require?  (rewritten against the conversation)
top 3     lubrication-spec  ·  lubrication-ariel  ·  lubrication-superior
HIT   the passage holding 'ISO VG 68' (lubrication-spec) is rank 1

answer: ISO VG 68
  ✓ Cooper-Bessemer GMV frames require a synthetic crankcase oil of grade ISO VG 68.
      cite lubrication-spec
```

The same words were typed. What was *searched* is different: the rewriter was shown the previous turn
and replaced "it" with what that turn established. The standalone question names its subject, so
retrieval can match it — and the GMV spec moves from rank 4 to rank **1**.

| | Searched | Rank of `lubrication-spec` | Answer |
|---|---|---|---|
| Memory off | *Which oil grade does it require?* | 4 — not in the reader's top 3 | not specified |
| Memory on | *Which oil grade does the Cooper-Bessemer GMV frame require?* | **1** | ISO VG 68 |

Everything after the rewrite is unchanged. The retriever, the reasoner and the proof tree received an
ordinary query; none of them knows a conversation exists.

## What the layer remembers

```text
conversation
 # │ via │ asked                               │ understood as                              │ answer
 1 │ ask │ Which frame is the TX-200 built on? │ —                                          │ ✓ Cooper-Bessemer GMV frame
 2 │ ask │ Which oil grade does it require?    │ Which oil grade does the Cooper-Bessemer   │ ✓ ISO VG 68
   │     │                                     │ GMV frame require?                         │
```

Each turn keeps both forms of the question: what you typed and what was actually run. A rewrite
changes what is searched and answered, so it is never hidden — this table is `TarnRag.history()`, and
the console prints `↳ understood as: …` after any `ask` or `retrieve` whose question was rewritten.

## What memory does not do

**It does not feed the conversation to the reader.** The reader is shown the standalone question and
the retrieved passages — no transcript, no earlier answer. Memory decides *which question is asked*;
it has no say in *what counts as evidence*. The answer `ISO VG 68` is cited to `lubrication-spec`
exactly as it would be if you had typed the full question yourself.

**It does not make a bad rewrite safe.** The rewrite is one model call, and a wrong one misleads the
whole turn — retrieval will faithfully fetch passages for the wrong question. That is why both forms
are recorded. In this example the rewriter is scripted (see
[what the scripted model is](index.md#what-the-scripted-model-is-and-is-not)), so this page shows that
a correct rewrite repairs retrieval; it does not measure how often a live model produces one.

**It is not free.** One extra model call per follow-up. None on a conversation's first turn, none with
memory off.

## Which operations take part

| Command | Reads the conversation | Records a turn |
|---|---|---|
| `ask` | yes | yes — the question and its answer |
| `retrieve` | yes | yes — the question only |
| `explain` | yes | **no** — a dry run |

`explain` rewrites the follow-up exactly as `retrieve` would and shows the result, but leaves the
conversation as it found it. That is what lets this example's `run.py` inspect each turn with
`explain` before asking it for real.

## Sessions

One console run, or one `TarnRag`, is one conversation. With `session_id: ""` each run starts a fresh
one. Turns are stored in the same SQLite file as the index and are kept until you delete them:

```text
tarn> history     the current conversation
tarn> new         start a fresh one (the old transcript stays in the store)
tarn> forget      delete the current conversation's turns
```

Set `session_id` to a name of your choosing to resume that conversation across runs.

## Run it

```bash
python -m examples.part_iii.example_02.run
```

Interactively, memory makes `retrieve` and `explain` conversational too — and resolving a follow-up
needs the live model named under `llm:` (`OPENAI_LLM_KEY`, `pip install '.[openai]'`):

```bash
python -m tarnrag.console examples/part_iii/example_02/config.yaml
tarn> ask Which frame is the TX-200 built on?
tarn> explain Which oil grade does it require?
tarn> ask Which oil grade does it require?
tarn> history
```

The console's opening lines state that memory is on and print the session id. Without a key, the
first `retrieve` of a conversation still works; a follow-up `retrieve` is searched as typed, with a
warning that names the missing key, and `ask` fails with the same message it always has.

## Where to go next

- [Enable conversation memory](../../how-to/enable-conversation-memory.md) — the recipe, for your own config.
- [Architecture overview](../../explanation/architecture.md#conversation-memory-sits-in-front-of-the-engines)
  — why the layer lives in the facade and why the reader never sees the conversation.

---

[← Part III](index.md) · [← A conversation without memory](01-a-conversation-without-memory.md)
