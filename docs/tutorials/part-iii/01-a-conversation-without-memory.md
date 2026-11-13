# 1 · A conversation without memory

> **Every question is treated as the first.**
> **Config:** `examples/part_iii/example_01/config.yaml` · **No API key needed** · builds the Part III store

This config is a plain single-hop setup — dense retrieval, a reader over the top 3 passages — with
the conversation layer present and switched **off**:

```yaml
  conversation_memory:
    class_name: conversation_memory
    enabled: false          # ← every question stands alone
```

Off is the library default, and it is exactly the behaviour of every earlier tutorial: nothing is
remembered, no extra model call is made.

## The first question stands on its own

```text
you › Which frame is the TX-200 built on?
searched  Which frame is the TX-200 built on?  (as typed)
top 3     compressor-models  ·  lubrication-spec  ·  lubrication-ariel
HIT   the passage holding 'GMV' (compressor-models) is rank 1

answer: Cooper-Bessemer GMV frame
  ✓ The TX-200 is a heavy-duty reciprocating compressor built on a Cooper-Bessemer GMV frame.
      cite compressor-models
```

The question names its subject (`TX-200`), so retrieval has something to match. Nothing here needs
memory.

## The follow-up does not

```text
you › Which oil grade does it require?
searched  Which oil grade does it require?  (as typed)
top 3     lubrication-ariel  ·  lubrication-superior  ·  lubrication-clark
MISS  the passage holding 'ISO VG 68' (lubrication-spec) is not in the top 3 — the reader is never shown it

answer: not specified
```

Look at what was searched: the question exactly as typed. "It" carries no information a retriever can
use, so the query is effectively *"which oil grade?"* — and this store has five frame lubrication specs
that answer that equally well. The three it returns are for the Ariel JGT (ISO VG 100), the Superior
MH (ISO VG 150) and the Clark HRA (ISO VG 32). The GMV spec, the one the conversation is about, is
rank 4.

The reader reads the top 3, so it never sees the GMV spec, and it says so: `not specified`.

Two things about that result are worth being precise about.

**The MISS involves no model.** It is a query embedding and a ranked list. Whatever reader you put
behind it — scripted, as here, or live — cannot read a passage it was not given.

**A deeper `top_k` does not fix it.** At `top_k: 8` the GMV spec is in the list, at rank 4 — behind
three other frames' specs and ahead of a fourth. A reader would then hold five different oil grades
and a question that does not say which frame it means. The problem is not how far down the list the
answer sits. The question itself is incomplete.

(The answer panel still says `grounded`. As in [Part II's rung 06](../part-ii/06-minimal-generation.md),
that only means no grounding check is configured — this config deliberately has none, so that the
conversation layer is the only thing that differs between the two examples.)

## Nothing was remembered

```text
what the conversation layer remembers
warning: conversation memory is disabled — set components.conversation_memory.enabled: true
```

With memory off there is no conversation to show. The facade says so rather than returning an empty
list that could be mistaken for "nothing asked yet".

## Run it

```bash
python -m examples.part_iii.example_01.run
```

Interactively, the retrieval half needs no key (`explain` never calls the reader):

```bash
python -m tarnrag.console examples/part_iii/example_01/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> ingest examples/docs/corpus-2-frames
tarn> status
tarn> explain Which oil grade does it require?
```

`status` reports **16 documents · 24 chunks**, and the `explain` shows the three wrong specs at the
top. `ask` uses the live model named under `llm:` in the config and needs its key (`OPENAI_LLM_KEY`,
`pip install '.[openai]'`).

## Next

[Example 02](02-conversation-memory.md) changes `enabled: false` to `true` and plays the same two
turns again.

---

[← Part III](index.md) · [Next: conversation memory →](02-conversation-memory.md)
