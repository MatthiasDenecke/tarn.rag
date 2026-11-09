# Use hosted embedding APIs

The default embedder runs a local ONNX model. To embed via a hosted API instead — OpenAI, Voyage,
or Gemini — switch the provider in `settings.embedding`.

## 1. Install and configure

```bash
pip install "tarn-rag[embeddings-api]"    # adds httpx (included in the `all` extra)
```

```bash
EMBEDDING__PROVIDER=openai                # or: voyage | gemini
EMBEDDING__MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536                  # MUST match the model's output dimension
```

The API key falls back to the provider's standard environment variable — `OPENAI_API_KEY`,
`VOYAGE_API_KEY`, or `GEMINI_API_KEY` — or set `EMBEDDING__API_KEY` explicitly.
`EMBEDDING__API_BASE_URL` overrides the endpoint (e.g. a proxy), `EMBEDDING__API_TIMEOUT` the
request timeout, and `EMBEDDING__BATCH_SIZE` the per-request batching.

## 2. Mind the index fingerprint

The same embedder embeds passages at ingest time and queries at search time — that identity is what
makes the vectors comparable. Its configuration (provider, model, dimension, prefixes, …) is
recorded as a fingerprint in the index's `index_meta`, and `RetrievalEngine` **refuses to open an
index whose fingerprint doesn't match the current settings**. Practically:

- Switching providers or models means **rebuilding the index** (re-ingesting into a fresh
  `database.document_url`). You cannot query an ONNX-built index through an API embedder or vice
  versa — the refusal is deliberate, not a bug.
- Keep one config per index. A config file pins the embedder and the store together, which is
  exactly what you want.

## Notes

- Ingestion throughput is now bounded by API latency and rate limits; tune
  `EMBEDDING__BATCH_SIZE` accordingly.
- For offline CI or demos, `EMBEDDING__PROVIDER=hash` is a deterministic, model-free embedder —
  no semantics, never for production.
- Asymmetric models that need instruction prefixes are supported via `EMBEDDING__QUERY_PREFIX` /
  `EMBEDDING__PASSAGE_PREFIX` — they are prepended to the text sent to the API exactly as for the local
  model (useful for an OpenAI-compatible endpoint serving E5 / BGE; Voyage and Gemini already distinguish
  queries from documents natively). Prefixes, `EMBEDDING__INJECT_HEADER_PATH`, and
  `EMBEDDING__CONTEXTUALIZE_TABLES` all join the fingerprint when set — see the
  [settings reference](../reference/settings.md).
