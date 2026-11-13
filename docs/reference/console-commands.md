# Console commands

Launch: `tarnrag <config>` (or `python -m tarnrag.console <config>`). Exactly one argument — the
config file (a `Settings` document; see the [settings reference](settings.md)). There is no
`--help` flag. Needs the `console` extra. A `.env` (searched from the current directory upward) is
loaded at startup; the config file stays authoritative and the environment only supplies what it
omits (e.g. `ANTHROPIC_API_KEY`).

| Command | Effect |
|---|---|
| `ingest <path> ...` | Ingest (or **re**-ingest) files; a directory ingests the files in it. The document id is the filename stem, so re-ingesting a file replaces it. |
| `docs` | List ingested documents (id, chunks, embeddings). Section-parent chunks are stored for auto-merge but not embedded, so a sectioned document shows fewer embeddings than chunks. |
| `status` | Summarize the corpus — counts + document-length stats. |
| `delete <id>` | Delete a document and everything derived from it. |
| `retrieve <query>` | Retrieval only — the ranked passages. |
| `explain <query>` | Retrieval with its inner workings — each retriever's candidates before fusion, the ranking at every pipeline stage (with component scores), and any routing. |
| `ask <query>` | Retrieval + generation — the grounded answer + its proof tree. |
| `help` | Show the commands. |
| `quit` | Exit (Ctrl-D / Ctrl-C also exit). |

↑/↓ give input history (readline). The console is purely a UI over the `TarnRag` facade — it owns
rendering and delegates all work to the facade's high-level methods.
