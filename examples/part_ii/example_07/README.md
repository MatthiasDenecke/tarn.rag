# Example 07 · Grounding check + abstain

Example 06 *trusted* the reader. This adds **verification** — the only delta:

```yaml
    grounding_checker: { class_name: llm_grounding }   # verify each claim against its cited evidence
    abstain: true                                      # refuse rather than return an unsupported answer
    min_grounded: 1.0                                  # every claim must be grounded
    refusal: "I don't have enough grounded evidence to answer that confidently."
```

Reads the **same base store**.

- **Verification (live).** The checker re-reads each claim against its evidence, so the proof's `✓` is
  now *earned* (Example 06 assumed it). The answerable question is still answered.
- **Abstain (constructed).** gpt-4o is honest — it says "not provided" rather than fabricating, so
  there's no reproducible way to make a *live* `ask` abstain. As the library's own grounding tests do,
  the script **plants** one unsupported claim (*"the pump carries a ten-year warranty"*, cited to a
  passage that never mentions a warranty) and runs it through `heuristic_grounding` (content-word
  overlap, deterministic). The check finds it disjoint from its evidence and the pipeline **refuses**.

📖 **[Tutorial: Grounding check + abstain](../../../docs/tutorials/part-ii/07-grounding-and-abstain.md)**
— what `min_grounded` really trades, and why the abstain demo is honestly labelled as constructed.

## Run

Needs `pip install '.[openai]'` and a key in **`OPENAI_LLM_KEY`** (or a repo-root `.env`).

```bash
python -m examples.part_ii.example_07.run    # the live verified answer, then the constructed abstain
```

...or interactively (the verified answer; the refusal needs the planted claim, so use the script):

```bash
python -m tarnrag.console examples/part_ii/example_07/config.yaml
tarn> ask how do I service a centrifugal pump before starting it?
```

→ Next: **[Example 08](../example_08)** — multi-hop: *"which oil grade does the TX-200 require?"* bridges
two documents (`compressor-models` → `lubrication-spec`), which one retrieve→read pass cannot reach.
