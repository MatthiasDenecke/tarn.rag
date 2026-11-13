# 6 · Minimal generation

> **Act B opens.** Passages become an answer — and the system cannot tell "I answered" from "I couldn't."
> **Config:** `examples/part_ii/example_06/config.yaml` · **Needs an LLM key** · reads the base store

Act A ended with good passages. Act B reads them.

The `generation_pipeline` — a `single_hop` reasoner plus a `provenance` assembler — was already pinned
in the base config (rung 00 specifies everything). This rung finally exercises it with `ask`. The only
config delta is the reader model:

```yaml
  llm:
    model: gpt-4o   # was gpt-4o-mini
```

`single_hop` does one retrieve-then-read pass and returns the answer plus a **proof tree**: which chunk
backed each claim.

## What you get

```text
tarn> ask how do I service a centrifugal pump before starting it?
```

The panel shows the answer — *"check the mechanical seal for leaks"* — and beneath it a proof step
marked `✓`, citing *Centrifugal pump maintenance*. That citation is the chunk the claim rests on. An
answer is never just a string here; it arrives with its evidence attached.

That is the good case, and it works.

## The limitation

```text
tarn> ask what is the maximum allowable working temperature of the cartridge mechanical seal?
```

The corpus *describes* the cartridge mechanical seal, but states no temperature anywhere. The reader
is honest about it and answers **"not provided"**.

And yet the panel header still reads **`answer (grounded)`**, and the pipeline does **not** abstain.

| | question | result |
|---|---|---|
| ✅ | *"How do I service a centrifugal pump before starting it?"* | *"check the mechanical seal for leaks"* — cites *Centrifugal pump maintenance* |
| ⚠️ | *"…maximum allowable working temperature of the cartridge mechanical seal?"* | *"not provided"* — **but still reported `grounded=True`, `abstained=False`** |

Sit with the second row, because it is the whole point of the rung. The system produced a non-answer
and **labelled it grounded**. Nothing in the output distinguishes "here is the answer, backed by
evidence" from "I could not answer this."

## Why: the pipeline trusts the reader

There is no verification step in this configuration. The reasoner reads the passages, writes an
answer, and the assembler attaches the passages it was given as citations. Nobody ever checks whether
the answer's claims are actually *supported* by those citations.

So `grounded=True` here does not mean "verified against the evidence". It means **"we did not check."**
The flag is an assumption wearing the costume of a fact.

A downstream consumer — a UI badge, an API client, an agent — cannot tell the difference. It will show
a green check on a non-answer.

## A note on the reader model

The config pins `gpt-4o` rather than `gpt-4o-mini` deliberately. `gpt-4o` is honest and extractive: it
answers when the passages support it, and says "not provided" when they don't. It does not fabricate.
(`gpt-4o-mini` intermittently *declined the answerable question*, even at temperature 0 — the opposite
failure, and a distracting one for a tutorial.)

That honesty is convenient here, but do not mistake it for a guarantee. It is a property of one model
on one corpus, and it is exactly the kind of thing you should not build a system's integrity on. Which
is what the next rung is about.

## Run it

```bash
pip install '.[openai]'          # the LLM backend
```

Put your key in **`OPENAI_LLM_KEY`** (exported, or in a repo-root `.env`, which the console and the
runner both load). Then:

```bash
python -m examples.part_ii.example_06.run
```

Or interactively:

```bash
python -m tarnrag.console examples/part_ii/example_06/config.yaml
tarn> ask how do I service a centrifugal pump before starting it?
tarn> ask what is the maximum allowable working temperature of the cartridge mechanical seal?
```

## Next

**[7 · Grounding check + abstain →](07-grounding-and-abstain.md)** — stop trusting the reader. Verify
every claim against its evidence, and refuse when it doesn't hold.

---

[← Part II](index.md)
