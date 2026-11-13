# 4 · Grounded generation

> **Question → an answer you can check, or a refusal you can trust.**
> **Code:** `examples/part_i/example_04/{ingestion.json, retrieval.json, generation.json, questions.json, ingestion.py, generation.py}`
> **Needs an LLM** for the generation step (ingestion and retrieval stay offline).

The first three tutorials end at a ranked list of passages. This one closes the loop: a reader model
turns passages into an answer, a grounding check verifies each claim against the evidence, and an
abstention policy refuses when the evidence doesn't hold.

## Run it

```bash
python -m examples.part_i.example_04.ingestion                            # offline
ANTHROPIC_API_KEY=sk-... python -m examples.part_i.example_04.generation
```

Without a key it still ingests and retrieves, and prints what it *would* send the model — so you can
follow along without spending anything. The default provider is Anthropic; the key itself never goes
in a config file (a config names the *env var*, not the secret).

## Everything composable comes from a file

Three specs sit next to the script, and the Python just loads them:

```python
settings = base_settings(
    db_path,
    components={RETRIEVAL_PIPELINE: RETRIEVAL_SPEC, GENERATION_PIPELINE: GENERATION_SPEC},
)

async with TarnRag(settings) as tarn:
    result = (await tarn.ask(question)).value
```

`retrieval.json` is the hybrid pipeline from tutorial 3 (dense + sparse, RRF-fused) — what fetches
the passages. `generation.json` is the new part:

```json
{
  "class_name": "generation_pipeline",
  "reasoner": {"class_name": "single_hop", "top_k": 4},
  "assembler": {"class_name": "provenance"},
  "grounding_checker": {
    "class_name": "cascading_grounding",
    "checkers": [
      {"class_name": "heuristic_grounding"},
      {"class_name": "llm_grounding"}
    ]
  },
  "min_grounded": 0.5,
  "abstain": true
}
```

Four decisions, each worth its own paragraph.

**`single_hop` reasoner.** One retrieval, one LLM call. The *default* reasoner is `decomposition`,
which splits a question into sub-questions, retrieves for each, and synthesizes — better on average
for multi-hop questions, but several LLM calls per answer. This example pins the cheap path
explicitly. Reasoners are components like everything else; swapping this one line changes the
strategy.

**`cascading_grounding`.** Grounding verifies each claim in the answer against the retrieved
passages. The cascade runs a cheap lexical-overlap heuristic *first*, and escalates only the claims
it is unsure about to the LLM. Claims that obviously restate a passage never cost a call; only the
genuinely ambiguous ones do. This is a cost/latency design, and it is the reason grounding is
affordable to run on every answer rather than as an offline audit.

**`min_grounded: 0.5` + `abstain: true`.** The abstention policy: if fewer than half the claims
survive grounding, refuse to answer. This is the knob that decides what the system does when it
doesn't know.

## Why abstention is the point

Recall tutorial 1: retrieval always returns its nearest neighbours. It has no notion of "nothing here
is relevant" — ask a three-document corpus about France and you will get three ranked hits about
pumps, tanks, and a marsupial.

The preview output makes this concrete. Even for a *good* question, look at what comes back:

```text
Passages retrieval would feed the model for:  How should I service a centrifugal pump before restarting it?
  [1] pump-maintenance: fy bearing lubrication, and confirm shaft alignment before restarting the pump.
  [2] pump-maintenance: Centrifugal pump maintenance: check the mechanical seal for leaks, verify bearing lubricat
  [3] tank-inspection: lement before returning the tank to service.
  [4] quokka: The quokka is a small macropod about the size of a domestic cat, native to Western Austral
```

The quokka is in the context window. It is irrelevant, it was retrieved anyway, and a model asked to
be helpful over that context is being handed an opportunity to say something false.

So `questions.json` includes a question the corpus cannot answer:

```json
[
  {"question": "How should I service a centrifugal pump before restarting it?"},
  {"question": "What detects corrosion in a storage tank?"},
  {"question": "What is the capital of France?"}
]
```

Retrieval will happily return passages for the third one. Grounding is what notices that none of them
support the answer, and abstention is what turns that into a refusal instead of a confident lie.
**That is the whole reason this layer exists.**

## What you get back

With a key set, each question prints an answer and its **proof tree** — every claim, whether grounding
found it supported, and the passages it cites:

```text
Q: How should I service a centrifugal pump before restarting it?
A: <the model's answer>    (grounded=True)
   - [grounded] <a claim the answer makes>
       cite: <header path, locator, or document id>
   - [grounded] <another claim>
       cite: ...

Q: What is the capital of France?
A: [abstained] <the refusal>
```

The exact wording depends on the model, but the *structure* does not: an answer is never just a
string. `GenerationResult` carries `answer`, `grounded`, `abstained`, and `proof` — a list of steps,
each with its claim, its grounding verdict, and its citations. A claim that no passage supports is
printed `[UNSUPPORTED]` and counts against `min_grounded`.

This is what "a proof tree" buys you: the answer is *auditable*. You can show a user which passage
each claim rests on, and you can find out — mechanically, not by vibes — when a claim rests on
nothing.

## Try it

- **Set `abstain` to `false`** and re-ask about the capital of France. The system now answers from a
  corpus that knows only about pumps, tanks, and quokkas. Read what it says. This is the single most
  instructive thing in Part I.
- **Raise `min_grounded` to `0.9`.** A much stricter system: it abstains unless nearly every claim is
  supported. Watch the good questions start to get refused too. Abstention is a precision/recall
  trade-off, not free safety.
- **Drop `llm_grounding` from the cascade**, leaving only the heuristic. Cheaper and faster; now watch
  which paraphrased-but-correct claims get marked `[UNSUPPORTED]` because they share no words with the
  passage that supports them.
- **Swap `single_hop` for `decomposition`** and ask something that needs two facts from two documents.

## Where to go next

Part I ends here: you can ingest a corpus, compose the pipeline that indexes it, choose and *measure*
a retrieval method, and generate an answer that carries its own evidence.

**[Part II →](../part-ii/index.md)** takes the next step, and it takes it the hard way: a ladder of
examples where each rung shows a query that works and a query that **fails**, and the next rung's
config repairs the failure. Dense retrieval misses an exact part number; hybrid finds it but demotes
a paraphrase; a reranker recovers the paraphrase; and so on. It is the same machinery you have just
learned, driven until it breaks.

Other directions:

- [Swap the retrieval pipeline](../../how-to/swap-the-retrieval-pipeline.md) — every retrieval
  composition, side by side.
- [Component catalog](../../reference/components.md) — every reasoner, grounding checker, chunker,
  and retriever you can name in a spec.
- [Architecture overview](../../explanation/architecture.md) — the three engines, the one store, and
  the invariants that hold them together.

---

[← Part I](index.md)
