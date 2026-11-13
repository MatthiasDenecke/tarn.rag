# Example 06 · Minimal generation (Act B opener)

Act A retrieved passages; **Act B reads them into an answer.** The `generation_pipeline` (single_hop
reasoner + provenance assembler) was already pinned in the base config — here we finally exercise it
with `ask`. The only delta is the reader model:

```yaml
  llm:
    model: gpt-4o   # was gpt-4o-mini — which intermittently declines answerable questions, even at temp 0
```

Reads the **same base store**. `single_hop` does one retrieve→read pass and returns the answer plus a
**proof tree** (which chunk backed each claim). The limitation is the point: the pipeline **trusts the
reader**. Ask something the corpus can't answer and you get a terse `not provided` — still reported
`grounded=True, abstained=False`. It cannot tell *"I answered"* from *"I couldn't."*

📖 **[Tutorial: Minimal generation](../../../docs/tutorials/part-ii/06-minimal-generation.md)** — why
`grounded=True` here means "we did not check".

## Run

Needs `pip install '.[openai]'` and a key in **`OPENAI_LLM_KEY`** (or a repo-root `.env`).

```bash
python -m examples.part_ii.example_06.run
```

...or interactively:

```bash
python -m tarnrag.console examples/part_ii/example_06/config.yaml
tarn> ask how do I service a centrifugal pump before starting it?
tarn> ask what is the maximum allowable working temperature of the cartridge mechanical seal?
```

→ Next: **[Example 07](../example_07)** — verify each claim, and refuse when it doesn't hold.
