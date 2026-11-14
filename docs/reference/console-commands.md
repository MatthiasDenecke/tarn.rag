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
| `history` | The current conversation — each turn as typed, as understood, and its answer. |
| `new` | Start a fresh conversation. The previous transcript stays in the store. |
| `forget` | Delete the current conversation's turns from the store. |
| `help` | Show the commands. |
| `quit` | Exit (Ctrl-D / Ctrl-C also exit). |

`history`, `new` and `forget` belong to [conversation memory](../how-to/enable-conversation-memory.md)
(`components.conversation_memory.enabled`); with it off they print a warning that memory is disabled.
With it on, the start-up banner shows the session id, one console run is one conversation, and
`retrieve` / `explain` / `ask` first rewrite a follow-up into a standalone question:

- after `ask` or `retrieve`, a rewritten question is shown as `↳ understood as: …`;
- `explain` shows the rewritten query with an `asked: …` line carrying the original wording. It is a
  dry run — it records no turn.

↑/↓ give input history (readline). The console is purely a UI over the `TarnRag` facade — it owns
rendering and delegates all work to the facade's high-level methods.
