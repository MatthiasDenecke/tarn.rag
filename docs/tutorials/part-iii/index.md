# Part III — conversations

[Part I](../part-i/index.md) and [Part II](../part-ii/index.md) ask one question at a time. Real use
rarely does. The second question leans on the first:

```text
you › Which frame is the TX-200 built on?
you › Which oil grade does it require?
```

"It" is the frame the first answer just named. Nothing in the second question says so — and a system
that treats every question as its first cannot know. Part III is about that gap, and the one config
line that closes it.

| # | Config | What happens to the follow-up |
|---|--------|-------------------------------|
| [01](01-a-conversation-without-memory.md) | `conversation_memory.enabled: false` | Searched as typed. The passage holding the answer is rank **4**; the reader, which reads the top 3, is never shown it. |
| [02](02-conversation-memory.md) | `conversation_memory.enabled: true` | Rewritten into a standalone question first. The same passage is rank **1**. |

The two configs differ in exactly that one line:

```bash
diff examples/part_iii/example_01/config.yaml examples/part_iii/example_02/config.yaml
```

## Setup

```bash
pip install -e ".[onnx,console]"   # the package, the ONNX embedding runtime, the console renderer
python scripts/fetch_model.py      # the embedding model (once)
```

**No API key is needed to run either example.** Each `run.py` injects a scripted, offline stand-in
for the LLM — see [what the scripted model is](#what-the-scripted-model-is-and-is-not) below. To drive
the same configs against a live model, use the console with a key; each page shows how.

## How Part III is built

It keeps Part II's conventions: each example **is** a fully explicit YAML config you can open in the
console, and its `run.py` only narrates. Two things are specific to this series.

**One store, shared.** Both examples read `examples/part_iii/store/conversation.db`, built from
`corpus-2` plus `corpus-2-frames` — **16 documents · 24 chunks**, the corpus
[Part II's rung 08](../part-ii/08-multi-hop.md) uses. It holds five near-identical frame lubrication
specs, which is what makes an unresolved "it" expensive: there are four wrong specs to retrieve
instead of the right one. Whichever example runs first builds the store; the other reuses it.

**The verdict is model-free.** Each turn prints where the passage holding the answer ranked, at the
depth the reader reads at. That check embeds a query and looks at the result list — no LLM is
involved, scripted or otherwise. It is the evidence these pages rest on.

## What the scripted model is, and is not

The examples have to run offline, so `run.py` passes `TarnRag` a `StaticLanguageModel` in place of a
live LLM (`examples/part_iii/_conversation.py`). A script can prove anything if you let it, so this
one is bound to what the pipeline actually hands it and knows nothing else:

- As the **rewriter**, it resolves "it" to the most recent answer *in the transcript it is shown*. If
  memory did not deliver the earlier turn, there is nothing to substitute and the question comes back
  unchanged.
- As the **reader**, it answers only from *the passages it is shown*, and only when one of them
  mentions what the question names. Otherwise it says `not specified`.

So the examples demonstrate two real things — the plumbing (store → window → rewrite → retrieval →
reader) and the retrieval difference a standalone question makes. They do **not** demonstrate how a
live model words a rewrite, or how often it gets one wrong. That needs a labelled set of follow-ups
scored through the eval harness, which this series does not include.

---

[← Tutorials](../index.md) · [← Part II](../part-ii/index.md) · [Start →](01-a-conversation-without-memory.md)
