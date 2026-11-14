# Enable conversation memory

By default every `ask`, `retrieve` and `explain` stands alone. Conversation memory makes a follow-up
question — *"which oil grade does it require?"* — resolve against the last few turns before it is
searched. It is one flag in the config. For a worked example, see
[Part III of the tutorials](../tutorials/part-iii/index.md).

## Turn it on

Add (or edit) the `conversation_memory` spec under `components`. YAML is shown; JSON works
identically.

```yaml
components:
  conversation_memory:
    class_name: conversation_memory
    enabled: true            # the switch — false (the default) is fully stateless
    session_id: ""           # "" = a fresh conversation per run; set an id to resume one
    max_turns: 6             # how many recent turns the rewriter is shown
    max_answer_chars: 600    # each remembered answer is cut to this first
    contextualizer:
      class_name: condense   # one LLM call that rewrites the follow-up
      max_tokens: 256
      temperature: 0.0       # pinned, so a rewrite is repeatable
```

Only `class_name` and `enabled: true` are required; the rest are shown at their defaults.

Memory needs an LLM for the rewrite — the same one generation uses (`llm:` in the config, key from
the environment). It is not called on a conversation's first turn.

## Use it

Nothing changes in how you call the facade. One `TarnRag` is one conversation:

```python
from tarnrag import TarnRag

async with TarnRag("config.yaml") as tarn:
    await tarn.ask("Which frame is the TX-200 built on?")
    answer = await tarn.ask("Which oil grade does it require?")   # resolved against the first turn

    for turn in (await tarn.history()).value:
        print(turn.question, "→", turn.standalone_question, "→", turn.answer)
```

`ask` and `retrieve` record a turn; `explain` resolves the follow-up the same way but records
nothing. In the console the same applies, plus `history`, `new` and `forget`
([console commands](../reference/console-commands.md)).

## See what a question was rewritten to

A rewrite changes what is searched, so it is always recoverable:

- `tarn.history()` returns each turn's `question` (as typed) and `standalone_question` (as run).
- `tarn.explain(...)` returns a trace whose `query.text` is the query as run; its `contextualized`
  annotation holds the original wording.
- The console prints `↳ understood as: …` after an `ask` or `retrieve` that was rewritten.

## Check the report

If a follow-up cannot be rewritten, the call carries on with the question as typed and says so in the
`Outcome`'s report — it does not fail and it does not stay silent:

```python
outcome = await tarn.retrieve("and its seals?")
for issue in outcome.report.issues:
    print(issue.severity.value, issue.message)
```

This happens when no LLM key is configured, or when the model's reply is not a usable question. An
error from the LLM call itself (network, authentication) is raised instead.

## Start over, or pick up where you left off

```python
await tarn.new_conversation()   # later calls no longer see the earlier turns; they stay stored
await tarn.forget()             # delete the current conversation's turns from the store
```

To resume a conversation in a later process, give it a name — in the config (`session_id: shift-a`)
or when constructing the facade, which takes precedence:

```python
async with TarnRag("config.yaml", session_id="shift-a") as tarn:
    ...
```

## Things to know

- **Turns are stored in the document store** (`conversation_turns`, the same SQLite file or Postgres
  database as the index) and are kept until `forget()`. Only the last `max_turns` are ever shown to
  the rewriter. A copied index file carries its conversations with it.
- **One facade, one conversation.** Unrelated questions asked through one memory-enabled `TarnRag`
  are read as one conversation. Call `new_conversation()` between them, or leave memory off for
  batch work.
- **The engines are unaffected.** `RetrievalEngine` and `GenerationEngine` used directly, and the
  eval harnesses that drive them, stay stateless. Memory is applied by `TarnRag`.
- **Cost:** one extra LLM call per follow-up turn.

## Turn it off

Set `enabled: false`, or remove the block. `history`, `new_conversation` and `forget` then return an
empty value with a warning that memory is disabled.

[← How-to guides](index.md)
