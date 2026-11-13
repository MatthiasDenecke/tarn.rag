"""Shared helpers for the example suite — imported by every example as ``examples.common``.

Examples are run as modules from the repo root, e.g.::

    python -m examples.part_i.example_01.ingestion

so this module imports cleanly as ``examples.common`` (no ``sys.path`` juggling). It holds the
incidental setup every example repeats — the model + docs locations, the model-present check, a
default ``Settings`` factory, and a per-example SQLite path. Each example keeps the *interesting*
part — the config knob or API it demonstrates — explicit and inline.
"""

from __future__ import annotations

import os
from pathlib import Path

from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings

EXAMPLES_DIR = Path(__file__).resolve().parent          # the examples/ package
REPO_ROOT = EXAMPLES_DIR.parent
DOCS_DIR = EXAMPLES_DIR / "docs"                        # holds named corpora (docs/corpus-1, ...)
MODEL_DIR = REPO_ROOT / "models" / "gte-small"          # local ONNX model + tokenizer


def corpus(name: str) -> Path:
    """Path to a named document corpus under ``examples/docs/`` (e.g. ``corpus("corpus-1")``).
    Add a corpus by dropping files in a new ``examples/docs/<name>/`` directory."""
    return DOCS_DIR / name


def require_model() -> None:
    """Exit with a friendly hint if the embedding model hasn't been fetched yet — every example
    needs it (to embed passages and queries)."""
    if not (MODEL_DIR / "model.onnx").exists():
        raise SystemExit(
            f"Embedding model not found under {MODEL_DIR}.\n"
            f"Fetch it once (needs network):  python scripts/fetch_model.py"
        )


def example_db(example_file: str) -> Path:
    """The SQLite store for ONE example: a ``rag_docs.db`` next to the calling script, so each
    example keeps its own isolated, re-runnable store. Pass ``__file__`` from the example.

    Set ``EXAMPLES_DATA_DIR`` to redirect the store out of the source tree (tests point it at a
    temp dir). The path is namespaced by the example's directory name so several examples can share
    one data dir without colliding. Call this inside ``main()`` so the override takes effect."""
    here = Path(example_file).resolve().parent
    data_dir = os.environ.get("EXAMPLES_DATA_DIR")
    return (Path(data_dir) / here.name if data_dir else here) / "rag_docs.db"


def load_env() -> None:
    """Load the repo-root ``.env`` into the process environment, if ``python-dotenv`` is installed — so a
    secret like the LLM key (e.g. ``OPENAI_LLM_KEY``) is available without exporting it by hand. The
    explicit YAML config still wins (pydantic-settings ranks init values above env vars), so this only
    supplies what a config doesn't set — chiefly the API key. A no-op when there's no ``.env`` / dotenv."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")  # override=False: a real shell env var beats the .env entry


def load_config(yaml_path: str | Path) -> Settings:
    """Load a Part II YAML config into ``Settings`` (the same file the console runs), redirecting the
    SQLite store into ``EXAMPLES_DATA_DIR`` when it is set — so a *test* run uses an isolated store while
    a human run writes to the path the config declares. The config is authoritative for everything else
    (embedding, pipelines), so the redirected store stays embedding-compatible with the original.
    Loads ``.env`` first so the LLM key is picked up (see :func:`load_env`)."""
    load_env()
    settings = Settings.from_file(yaml_path)
    data_dir = os.environ.get("EXAMPLES_DATA_DIR")
    if data_dir:
        db_name = Path(settings.database.document_url.split("///")[-1]).name
        settings.database.document_url = f"sqlite:///{Path(data_dir) / db_name}"
    return settings


def base_settings(db_path: Path, **overrides: object) -> Settings:
    """Default embedded/SQLite settings shared by every example.

    Pass ``db_path`` (usually ``example_db(__file__)``) and override only the knob an example is
    demonstrating, e.g. ``base_settings(db, components={INGESTION_PIPELINE: {...}})``. The default
    ``ID_POLICY='caller'`` lets examples use readable, stable document ids. An ingestion and a
    retrieval built from the same ``db_path`` + ``embedding`` stay compatible (retrieval validates
    an embedding *fingerprint*), so reuse this for both sides of an example.
    """
    defaults: dict[str, object] = {
        "_env_file": None,           # explicit config only — ignore any ambient .env, for reproducibility
        "MODE": "embedded",          # whole pipeline in-process (no Postgres / job queue)
        "ID_POLICY": "caller",       # examples supply their own stable document ids
        "EMBEDDING_DIMENSION": 384,  # output dimension of gte-small (the default model)
        "database": DatabaseSettings(document_url=f"sqlite:///{db_path}"),
        "embedding": EmbeddingSettings(model_dir=str(MODEL_DIR)),
    }
    return Settings(**{**defaults, **overrides})
