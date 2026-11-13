"""The shared embedding pipeline — used by ingestion (passages) and retrieval (queries).

Writing it once is what guarantees the §5.3 requirement that retrieval replays *exactly* the
pipeline ingestion used. The backend is **pluggable** behind the ``Embedder`` port (a ``Resource``):
``OnnxEmbedder`` (local ONNX, the default — prefix → tokenize → ONNX → pool → normalize) and the HTTP
API embedders (OpenAI / Voyage / Gemini, in ``embedder_api``). ``Embedder.create`` selects one from
``EmbeddingSettings.provider``; the embedder's identity (provider/model/dim/… for an API backend;
model/tokenizer/pooling/… for ONNX) is summarized by ``config_fingerprint()`` and recorded in
``index_meta``, so retrieval refuses to ``open()`` an index built by a different embedder.

Heavy deps (``onnxruntime`` / ``tokenizers`` for ONNX; ``httpx`` for the API backends) are imported
lazily, so this module — and ``config_fingerprint()`` / ``embed_meta()`` — work without them. Tests use
a fake ``Embedder``.
"""

from __future__ import annotations

import json
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Any

from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.core.hashing import sha256_file, sha256_hex
from tarnrag.core.resources.resource import Resource


class Embedder(Resource):
    """
    Port for the shared embedding pipeline — the seam that lets ingestion and retrieval use one
    model (and lets tests swap in a fake). A ``Resource`` (an injected model, not a ``Component``).
    """

    @staticmethod
    def create(embedding: EmbeddingSettings, embedding_dimension: int) -> Embedder:
        """Build the configured embedder from the embedding settings slice + the cross-cutting
        ``embedding_dimension`` — the provider-dispatching factory (the embedder analog of
        ``LanguageModel.create``). Local ``OnnxEmbedder`` by default; the HTTP API providers (``openai``
        / ``voyage`` / ``gemini``) are lazily imported (the ``embeddings-api`` extra). Every construction
        site — the engines, the facade, and the Embed stage — goes through here, so a pipeline spec and
        the retrieval side can't disagree on the backend. (Each provider class keeps its own concrete
        ``create``; this dispatches to it.)"""
        if embedding.provider == "onnx":
            return OnnxEmbedder.create(embedding, embedding_dimension)
        if embedding.provider == "hash":
            return HashEmbedder.create(embedding, embedding_dimension)
        from tarnrag.core.resources.embedder_api import EMBEDDER_PROVIDERS

        try:
            impl = EMBEDDER_PROVIDERS[embedding.provider]
        except KeyError:
            raise ValueError(
                f"unknown embedding provider {embedding.provider!r}; "
                f"choose 'onnx' or one of {sorted(EMBEDDER_PROVIDERS)}"
            ) from None
        return impl.create(embedding, embedding_dimension)

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (ingestion side) — uses the passage prefix."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (retrieval side) — uses the query prefix."""

    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def config_fingerprint(self) -> str: ...

    @abstractmethod
    def embed_meta(self) -> dict[str, str]:
        """The ``index_meta`` embedding keys (incl. ``embedding_config_fingerprint``)."""

    @staticmethod
    def _fingerprint(identity: dict[str, Any]) -> str:
        """Stable sha256 of an embedder's identity dict — the ``embedding_config_fingerprint`` recorded
        in ``index_meta`` and validated at ``open()``. Shared by every backend so the gate is uniform."""
        blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return sha256_hex(blob)


class OnnxEmbedder(Embedder):
    """
    ONNX Runtime embedder (prefix → tokenize → ONNX → mean-pool → L2). Heavy deps load lazily on
    the first ``embed``; the identity methods need only the tokenizer file, not the model.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        model_id: str,
        revision: str = "",
        embedding_dim: int = 384,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: str = "l2",
        query_prefix: str = "",
        passage_prefix: str = "",
        inject_header_path: bool = False,
        contextualize_tables: bool = False,
        model_file: str = "model.onnx",
        tokenizer_file: str = "tokenizer.json",
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.revision = revision
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.inject_header_path = inject_header_path
        self.contextualize_tables = contextualize_tables
        self.model_path = self.model_dir / model_file
        self.tokenizer_path = self.model_dir / tokenizer_file
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] | None = None
        # Ingestion stages run in worker threads and queries embed off-loop too, so two first calls can
        # race the lazy load; the lock makes it build the (large) session exactly once.
        self._load_lock = threading.Lock()

    @classmethod
    def create(cls, embedding: EmbeddingSettings, embedding_dimension: int) -> OnnxEmbedder:
        """Build the shared ONNX embedder from its config slice (model loads lazily on first
        ``embed``). Ingestion and retrieval both go through here, so they share one pipeline.
        ``embedding_dimension`` is passed separately — it's cross-cutting (index + repo match it)."""
        return cls(
            embedding.model_dir,
            model_id=embedding.model,
            revision=embedding.revision,
            embedding_dim=embedding_dimension,
            max_length=embedding.max_seq_length,
            pooling=embedding.pooling,
            normalize=embedding.normalize,
            query_prefix=embedding.query_prefix,
            passage_prefix=embedding.passage_prefix,
            inject_header_path=embedding.inject_header_path,
            contextualize_tables=embedding.contextualize_tables,
        )

    # ---------------- identity (no model load needed) ----------------

    def dim(self) -> int:
        return self.embedding_dim

    def identity(self) -> str:
        """The ``Resource`` identity — the model id (for provenance / eval / logging). Distinct from
        ``config_fingerprint``, which hashes the full pipeline identity to gate index compatibility."""
        return f"{self.model_id}@{self.revision}" if self.revision else self.model_id

    def _tokenizer_sha256(self) -> str:
        return sha256_file(self.tokenizer_path)

    def _identity(self) -> dict[str, Any]:
        """
        The embedding-pipeline identity — the single source of truth for both
        ``config_fingerprint`` (which hashes it) and ``embed_meta`` (which records it). The keys
        and value types here are FROZEN: they are exactly what gets hashed, so changing any of
        them changes every index's fingerprint. Add a new identity-bearing field here once.
        """
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "embedding_dim": self.embedding_dim,
            "tokenizer_sha256": self._tokenizer_sha256(),
            "pooling": self.pooling,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": self.max_length,
            "inject_header_path": self.inject_header_path,
            "contextualize_tables": self.contextualize_tables,
        }

    def config_fingerprint(self) -> str:
        """
        Stable hash of the pipeline identity (model/tokenizer/pooling/normalize/prefixes/…),
        recorded in ``index_meta`` — retrieval refuses to open an index whose fingerprint differs.
        """
        return self._fingerprint(self._identity())

    def embed_meta(self) -> dict[str, str]:
        """The ``index_meta`` embedding keys (incl. ``embedding_config_fingerprint``): the
        ``_identity`` values stringified, with model id/revision under their ``embedding_*``
        column names."""
        ident = self._identity()
        return {
            "embedding_model_id": ident["model_id"],
            "embedding_model_revision": ident["revision"],
            "embedding_dim": str(ident["embedding_dim"]),
            "tokenizer_sha256": ident["tokenizer_sha256"],
            "pooling": ident["pooling"],
            "normalize": ident["normalize"],
            "query_prefix": ident["query_prefix"],
            "passage_prefix": ident["passage_prefix"],
            "max_length": str(ident["max_length"]),
            "inject_header_path": str(ident["inject_header_path"]),
            "contextualize_tables": str(ident["contextualize_tables"]),
            "embedding_config_fingerprint": self.config_fingerprint(),
        }

    # ---------------- inference ----------------

    def _load(self):
        """
        Lazily build the ONNX session + tokenizer on first use (keeps import + identity cheap).
        """
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:  # another thread finished the load while we waited
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.tokenizer_path))
            tok.enable_truncation(self.max_length)
            tok.enable_padding()
            session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self._tokenizer = tok
            self._input_names = {i.name for i in session.get_inputs()}
            self._session = session  # published last: a non-None session means everything is ready

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        """
        Run one batch through the pipeline: tokenize (with ``prefix``) → ONNX forward →
        pool (``self.pooling``) → normalize (``self.normalize``). One vector per input.
        """
        if not texts:
            return []
        self._load()
        import numpy as np

        encs = self._tokenizer.encode_batch([prefix + t for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        out = self._session.run(None, feed)[0]  # [B, L, dim]
        return self._normalize(self._pool(out, mask)).tolist()

    def _pool(self, out, mask):
        """Pool the token outputs ``[B, L, dim]`` to ``[B, dim]`` per ``self.pooling``: ``mean`` over the
        attention mask (encoder models), ``last`` non-pad token (decoder embedders, e.g. Qwen), or
        ``cls`` (first token)."""
        import numpy as np

        if self.pooling == "mean":
            m = mask[..., None].astype(np.float32)
            return (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        if self.pooling == "cls":
            return out[:, 0]
        if self.pooling == "last":  # last non-pad token (right-padded ⇒ index = count(mask) − 1)
            last = mask.sum(axis=1).astype(np.int64) - 1
            return out[np.arange(out.shape[0]), last]
        raise ValueError(f"unknown pooling {self.pooling!r} (use 'mean', 'last', or 'cls')")

    def _normalize(self, pooled):
        """Normalize pooled vectors per ``self.normalize``: ``l2`` (unit length — cosine-equivalent KNN)
        or ``none``."""
        import numpy as np

        if self.normalize == "l2":
            norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            return (pooled / norms).astype(np.float32)
        if self.normalize == "none":
            return pooled.astype(np.float32)
        raise ValueError(f"unknown normalize {self.normalize!r} (use 'l2' or 'none')")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.passage_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.query_prefix)[0]


class HashEmbedder(Embedder):
    """A deterministic, dependency-free embedder — the embedding analog of ``StaticLanguageModel``: maps text
    to a fixed-dim vector with the hashing trick (each whitespace token hashed into a bucket), L2-normalized.

    Retrieval quality is poor (no semantics) — it exists so the **whole pipeline runs offline with no model
    download** (CI, demos, the eval loop), the way ``StaticLanguageModel`` lets generation run with no key.
    Selected via ``EmbeddingSettings.provider='hash'``; never for production retrieval."""

    PROVIDER = "hash"

    def __init__(self, *, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    @classmethod
    def create(cls, embedding: EmbeddingSettings, embedding_dimension: int) -> HashEmbedder:
        return cls(embedding_dim=embedding_dimension)

    def dim(self) -> int:
        return self.embedding_dim

    def identity(self) -> str:
        return f"{self.PROVIDER}:{self.embedding_dim}"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in text.lower().split():
            vec[int(sha256_hex(token)[:8], 16) % self.embedding_dim] += 1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def _identity(self) -> dict[str, Any]:
        return {"provider": self.PROVIDER, "embedding_dim": self.embedding_dim, "normalize": "l2"}

    def config_fingerprint(self) -> str:
        return self._fingerprint(self._identity())

    def embed_meta(self) -> dict[str, str]:
        return {
            "embedding_provider": self.PROVIDER,
            "embedding_model_id": self.PROVIDER,
            "embedding_dim": str(self.embedding_dim),
            "normalize": "l2",
            "embedding_config_fingerprint": self.config_fingerprint(),
        }
