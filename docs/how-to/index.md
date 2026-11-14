# How-to guides

Task-oriented recipes: each assumes you know *what* you want and shows the shortest correct path to
it. If you're new to tarn.rag, start with the [tutorials](../tutorials/index.md) instead.

- [Run distributed mode](run-distributed-mode.md) — Postgres + pgQueuer workers instead of the
  in-process default.
- [Swap the retrieval pipeline](swap-the-retrieval-pipeline.md) — dense-only, hybrid, reranked, or
  query-routed retrieval by editing one component spec.
- [Use hosted embedding APIs](use-hosted-embeddings.md) — OpenAI / Voyage / Gemini embedders in
  place of the local ONNX model.
- [Enable conversation memory](enable-conversation-memory.md) — make follow-up questions resolve
  against the last few turns, with one config flag.

Every knob these guides touch is documented exhaustively in the [reference](../reference/index.md).

[← Documentation index](../index.md)
