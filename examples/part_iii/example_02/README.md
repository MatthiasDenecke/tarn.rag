# Example 02 · The same conversation with memory

One line changes from Example 01:

```yaml
  conversation_memory:
    class_name: conversation_memory
    enabled: true           # ← was false
```

```bash
diff examples/part_iii/example_01/config.yaml examples/part_iii/example_02/config.yaml
```

The same two turns. Before the follow-up *"Which oil grade does it require?"* is searched, it is
rewritten against the previous turn into *"Which oil grade does the Cooper-Bessemer GMV frame
require?"*. The GMV spec moves from rank 4 to **rank 1** and the answer is `ISO VG 68`, cited to
`lubrication-spec`.

Retrieval, the reasoner and the proof tree are unchanged, and the conversation is never shown to the
reader. Each turn is stored with both forms of its question — as typed and as understood.

📖 **[Tutorial: Conversation memory](../../../docs/tutorials/part-iii/02-conversation-memory.md)** —
what memory does not do, which commands record a turn, and sessions.

## Run

No API key needed: `run.py` injects a scripted, offline stand-in for the LLM, so the rewrite shown is
scripted (it resolves "it" from the transcript it is handed, nothing more).

```bash
python -m examples.part_iii.example_02.run
```

...or interactively, against the live model under `llm:` (`OPENAI_LLM_KEY`, `pip install '.[openai]'`).
Reads the store Example 01 builds — ingest first if you start here:

```bash
python -m tarnrag.console examples/part_iii/example_02/config.yaml
tarn> ask Which frame is the TX-200 built on?
tarn> explain Which oil grade does it require?       # a dry run: shows the rewrite, records nothing
tarn> ask Which oil grade does it require?
tarn> history                                        # each turn as typed and as understood
```
