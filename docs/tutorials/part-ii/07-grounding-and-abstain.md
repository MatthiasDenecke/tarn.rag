# 7 · Grounding check + abstain

> **Stop trusting the reader.** Verify each claim against its evidence — and refuse when it doesn't hold.
> **Config:** `examples/part_ii/example_07/config.yaml` · **Needs an LLM key** · reads the base store

[Rung 06](06-minimal-generation.md) *assumed* its answers were grounded. This rung **checks**.

```yaml
    grounding_checker: { class_name: llm_grounding }   # verify each claim against its cited evidence
    abstain: true                                      # refuse rather than return an unsupported answer
    min_grounded: 1.0                                  # every claim must be grounded
    refusal: "I don't have enough grounded evidence to answer that confidently."
```

Two things arrive together, and they are separable ideas:

- **Verification.** A `grounding_checker` re-reads each claim against its cited evidence and stamps it
  grounded or ungrounded. The `✓` in the proof tree is now **earned** rather than assumed.
- **Abstention.** If fewer than `min_grounded` of the claims survive, the pipeline returns the
  `refusal` string instead of the answer.

Verification without abstention gives you an honest label on a bad answer. Abstention without
verification is impossible. You want both.

## The answerable question still answers

```text
tarn> ask how do I service a centrifugal pump before starting it?
```

You get the same answer as rung 06 — *"check the mechanical seal for leaks"*, citing *Centrifugal pump
maintenance*.

The difference is invisible and it is the entire point: `llm_grounding` ran a **second pass** to verify
that claim against the passage before the answer was returned. Same output, categorically different
epistemic status. Rung 06's `✓` was a guess; this one is a check.

Verification's success case is supposed to look like nothing happened. That is what makes it easy to
undervalue.

## Making the refusal visible

Now the awkward part, and the example is honest about it.

`gpt-4o` **will not fabricate** on this corpus. Ask it something the passages don't support and it says
"not provided" rather than inventing a number. So there is no reproducible way to make a *live* `ask`
abstain — the reader simply refuses to produce the unsupported claim that abstention exists to catch.

Rather than fake it with a prompt trick, the script does what the library's own grounding tests do: it
**plants** one unsupported claim — *"the pump carries a ten-year warranty"* — cited to a maintenance
passage that says nothing whatsoever about a warranty. It runs that through a grounding + abstain
pipeline using `heuristic_grounding` (content-word overlap, no LLM, fully deterministic).

The checker finds the claim disjoint from its evidence. `min_grounded: 1.0` is not met. The pipeline
**refuses**, returning the refusal string in place of the answer.

So: the verification half is demonstrated **live**, and the abstention half is demonstrated
**constructed**. The README and the script both say so plainly, and that is the right call — a tutorial
that manufactured a hallucination to show off its hallucination detector would be teaching you
something false about how often this fires.

## Why `min_grounded: 1.0`

Every claim must be grounded, or the whole answer is withheld. That is the strictest possible setting,
and it is a *choice*, not a default worth inheriting.

Abstention is a precision/recall trade, and this dial is where you make it:

- **`1.0`** — refuse unless everything checks out. Maximum trust in what you do say; you will refuse
  answers that were mostly right.
- **`0.5`** (what [Part I's example 04](../part-i/04-grounded-generation.md) uses) — answer if half the
  claims hold. More answers, weaker guarantee.

There is no correct value. There is only the question of what your users would rather have when the
system is unsure — a partial answer or an honest refusal — and that depends entirely on what they do
with it. A maintenance engineer following a procedure and a person browsing a knowledge base want
opposite things.

## Run it

```bash
pip install '.[openai]'   # key in OPENAI_LLM_KEY, or a repo-root .env
python -m examples.part_ii.example_07.run
```

Prints the live verified answer, then the constructed abstain — the refusal replacing the fabricated
answer.

Interactively, you can see the verified answer (the refusal needs the planted claim, so use the
script for that half):

```bash
python -m tarnrag.console examples/part_ii/example_07/config.yaml
tarn> ask how do I service a centrifugal pump before starting it?
```

## Where the ladder stands

You now have a system that retrieves well *and* knows the limits of what it retrieved:

- **Act A** — dense for meaning, sparse for exact terms, a reranker to correct both, structure-aware
  chunks so whole sections are retrievable, and a router to pick the method per query.
- **Act B** — a reader that answers with citations, a checker that verifies each claim against them,
  and a policy that refuses rather than guesses.

**The next failure is already in the corpus.** `compressor-models.md` says the `TX-200` is a
Cooper-Bessemer GMV frame; `lubrication-spec.md` says a GMV frame takes `ISO VG 68`. Ask *"which oil
grade does the TX-200 require?"* and no single retrieve-and-read pass can answer it — the bridge spans
two documents, and neither contains the whole chain.

**[8 · Multi-hop reasoning →](08-multi-hop.md)** — a reasoner that decomposes the question and
retrieves again, so the second hop reaches what the first could not.

---

[← Part II](index.md) · [Back to the tutorials index](../index.md)
