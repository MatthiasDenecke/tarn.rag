# Example 01 · A conversation without memory

The conversation layer is present and switched **off** — the library default:

```yaml
  conversation_memory:
    class_name: conversation_memory
    enabled: false          # ← every question stands alone
```

Two turns: *"Which frame is the TX-200 built on?"* then *"Which oil grade does it require?"*. The
follow-up is searched **as typed**. "It" names nothing, and this store holds five near-identical
frame lubrication specs — so the top 3 are other frames' specs, the GMV spec is **rank 4**, and the
reader (which reads the top 3) answers `not specified`.

The MISS is printed by a check that involves **no model**.

📖 **[Tutorial: A conversation without memory](../../../docs/tutorials/part-iii/01-a-conversation-without-memory.md)**
— why a deeper `top_k` would not fix it.

## Run

No API key needed: `run.py` injects a scripted, offline stand-in for the LLM.

```bash
python -m examples.part_iii.example_01.run
```

...or interactively. `explain` needs no key; `ask` uses the live model under `llm:` (`OPENAI_LLM_KEY`,
`pip install '.[openai]'`):

```bash
python -m tarnrag.console examples/part_iii/example_01/config.yaml
tarn> ingest examples/docs/corpus-2
tarn> ingest examples/docs/corpus-2-frames
tarn> status                                         # 16 documents · 24 chunks
tarn> explain Which oil grade does it require?       # three wrong specs on top
```

→ Next: **[Example 02](../example_02)** — one config line, and the follow-up is rewritten first.
